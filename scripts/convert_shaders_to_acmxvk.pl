#!/usr/bin/env perl

use strict;
use warnings;

use File::Basename qw(dirname);
use File::Find qw(find);
use File::Path qw(make_path);
use File::Spec;
use FindBin qw($Bin);
use Getopt::Long qw(GetOptions);
use IPC::Open3;
use JSON::PP;
use Symbol qw(gensym);

my $input = File::Spec->catdir($Bin, '..', 'shaders_new');
my $compute_input = File::Spec->catdir($Bin, '..', 'compute');
my $output = File::Spec->catdir($Bin, '..', 'shaders_acmxvk');
my $glslc = 'glslc';
my $force = 0;
my $dry_run = 0;
my $limit = 0;
my $help = 0;

GetOptions(
    'input=s'         => \$input,
    'compute-input=s' => \$compute_input,
    'output=s'        => \$output,
    'glslc=s'         => \$glslc,
    'force'           => \$force,
    'dry-run'         => \$dry_run,
    'limit=i'         => \$limit,
    'help'            => \$help,
) or usage(1);

usage(0) if $help;
die "--limit must not be negative\n" if $limit < 0;

$input = expand_tilde($input);
$compute_input = expand_tilde($compute_input);
$output = expand_tilde($output);
$glslc = expand_tilde($glslc);

die "Input directory does not exist: $input\n" if !-d $input;
die "Compute input directory does not exist: $compute_input\n"
  if !-d $compute_input;
die "Input and output directories must be different\n"
  if File::Spec->rel2abs($input) eq File::Spec->rel2abs($output);
die "Compute input and output directories must be different\n"
  if File::Spec->rel2abs($compute_input) eq File::Spec->rel2abs($output);

my ($manifest_order, $custom_metadata) = read_input_manifest($input);
my ($compute_order, $compute_metadata) = read_input_manifest($compute_input);
for my $name (keys %{$compute_metadata}) {
    $custom_metadata->{$name} = $compute_metadata->{$name}
      if !exists $custom_metadata->{$name};
}
my @fragment_paths = discover_shaders($input, $manifest_order, 'frag');
my @compute_paths = discover_shaders($compute_input, $compute_order, 'comp');
my @shader_entries = (
    (map { { root => $input, relative => $_, stage => 'frag' } }
       @fragment_paths),
    (map { { root => $compute_input, relative => $_, stage => 'comp' } }
       @compute_paths),
);
splice @shader_entries, $limit if $limit && @shader_entries > $limit;

die "No fragment or compute shader sources found in $input\n"
  if !@shader_entries;

my %builtin_uniform = map { $_ => 1 } qw(
  alpha amp amp_high amp_low amp_mid amp_peak amp_rms amp_smooth history_head
  iFrame iFrameRate iMouse iMouseClick iResolution iSampleRate iTime
  iTimeDelta iamp spectrum_history_head spectrum_history_size time_f uamp
);

my @custom_names = manifest_custom_names($custom_metadata);
my %custom_seen = map { $_ => 1 } @custom_names;
for my $entry (@shader_entries) {
    my $source = read_text(
        File::Spec->catfile($entry->{root}, split m{/}, $entry->{relative})
    );
    while ($source =~ /^\s*uniform\s+float\s+([^;]+);/gm) {
        for my $declaration (split /,/, $1) {
            my ($name) = $declaration =~ /^\s*([A-Za-z_][A-Za-z0-9_]*)/;
            next if !defined $name || $builtin_uniform{$name} || $custom_seen{$name};
            push @custom_names, $name;
            $custom_seen{$name} = 1;
        }
    }
}

if (@custom_names > 64) {
    warn "Found " . scalar(@custom_names)
      . " custom float uniforms; ACMXVK supports 64. Extra names will fail conversion.\n";
    splice @custom_names, 64;
}

my %custom_slot;
for my $index (0 .. $#custom_names) {
    $custom_slot{$custom_names[$index]} = $index;
}

if (!$dry_run) {
    make_path($output) if !-d $output;
    die "Cannot write output directory: $output\n" if !-w $output;
    find_program($glslc);
}

my @converted;
my @report;
my $failed = 0;
my $history_count = 0;
my $manifest_omitted = 0;

for my $entry (@shader_entries) {
    my $relative = $entry->{relative};
    my $input_path =
      File::Spec->catfile($entry->{root}, split m{/}, $relative);
    my $stage = $entry->{stage};
    my $source = read_text($input_path);
    my ($converted_source, $warnings) =
      convert_source($source, $stage, \%custom_slot);

    my $output_relative = output_source_name($relative, $stage);
    $output_relative = "compute/$output_relative" if $stage eq 'comp';
    my $spirv_relative = "$output_relative.spv";
    my $output_path = File::Spec->catfile($output, split m{/}, $output_relative);
    my $spirv_path = File::Spec->catfile($output, split m{/}, $spirv_relative);

    ++$history_count if grep { /history binding 2/ } @{$warnings};

    if ($dry_run) {
        print "would convert $relative -> $spirv_relative\n";
        push @report, report_entry($relative, 'dry-run', $warnings, '');
        next;
    }

    if (!$force && -f $output_path && -f $spirv_path) {
        my $uses_history = grep { /history binding 2/ } @{$warnings};
        if ($uses_history) {
            ++$manifest_omitted;
            push @report,
              report_entry($relative, 'existing-not-listed', $warnings,
                           'existing SPIR-V omitted from library.json');
        } else {
            push @converted, $spirv_relative;
            push @report, report_entry($relative, 'existing', $warnings, '');
        }
        print "kept $spirv_relative\n";
        next;
    }

    if (!$force && (-e $output_path || -e $spirv_path)) {
        ++$failed;
        my $message = 'incomplete output exists (use --force to replace it)';
        warn "skip $relative: $message\n";
        push @report, report_entry($relative, 'skipped', $warnings, $message);
        next;
    }

    make_path(dirname($output_path));
    write_text($output_path, $converted_source);

    my ($status, $compiler_output) = run_compiler(
        $glslc, $stage, $output_path, $spirv_path
    );
    if ($status != 0) {
        ++$failed;
        unlink $spirv_path if -e $spirv_path;
        warn "failed $relative\n";
        push @report,
          report_entry($relative, 'failed', $warnings, $compiler_output);
        next;
    }

    my $uses_history = grep { /history binding 2/ } @{$warnings};
    if ($uses_history) {
        ++$manifest_omitted;
        push @report,
          report_entry($relative, 'compiled-not-listed', $warnings,
                       'SPIR-V was generated but omitted from library.json');
    } else {
        push @converted, $spirv_relative;
        push @report, report_entry($relative, 'converted', $warnings, '');
    }
    print "converted $relative -> $spirv_relative\n";
}

if (!$dry_run) {
    my %output_custom;
    for my $name (@custom_names) {
        my $metadata = $custom_metadata->{$name};
        $output_custom{$name} = ref($metadata) eq 'HASH'
          ? normalize_custom_metadata($metadata)
          : {
                minimum => 0.0,
                maximum => 1.0,
                step    => 0.01,
                value   => 0.0,
            };
    }

    my $manifest = {
        version         => 1,
        custom_uniforms => \%output_custom,
        shaders         => \@converted,
    };
    my $json = JSON::PP->new->canonical->pretty->encode($manifest);
    write_text(File::Spec->catfile($output, 'library.json'), $json);
    write_text(
        File::Spec->catfile($output, 'conversion-report.txt'),
        join("\n", @report) . "\n"
    );
}

my $verb = $dry_run ? 'would inspect' : 'inspected';
print "$verb " . scalar(@shader_entries) . " shader(s); ";
print scalar(@converted) . " listed, $failed failed or skipped";
print ", $history_count use history binding 2" if $history_count;
print ", $manifest_omitted compiled but omitted from library.json"
  if $manifest_omitted;
print "\n";
print "Output: $output\n" if !$dry_run;

exit($failed ? 2 : 0);

sub usage {
    my ($status) = @_;
    print <<'USAGE';
Convert ACMX2/OpenGL shaders into ACMXVK Vulkan GLSL and SPIR-V.

Usage:
  perl scripts/convert_shaders_to_acmxvk.pl [options]

Options:
  --input DIR          Fragment source directory (default: shaders_new)
  --compute-input DIR  Compute source directory (default: compute)
  --output DIR         Destination directory (default: shaders_acmxvk)
  --glslc PATH         Vulkan GLSL compiler (default: glslc from PATH)
  --force              Replace previously generated shader and SPIR-V files
  --dry-run            List candidate shaders without writing or compiling
  --limit N            Convert only the first N shaders (useful for testing)
  --help               Show this help

The converter never modifies the input directory. It writes converted .frag or
.comp source, compiled .spv files, library.json, and conversion-report.txt.
Only successfully compiled modules are placed in library.json.
History-binding modules are compiled but omitted because ordinary ACMXVK
post-processing passes do not expose binding 2 yet.
USAGE
    exit $status;
}

sub expand_tilde {
    my ($path) = @_;
    return $path if $path !~ m{^~(?:/|$)};
    die "HOME is not set; cannot expand $path\n" if !defined $ENV{HOME};
    $path =~ s{^~(?=/|$)}{$ENV{HOME}};
    return $path;
}

sub find_program {
    my ($program) = @_;
    return if File::Spec->file_name_is_absolute($program) && -x $program;
    return if $program =~ m{/} && -x $program;
    for my $directory (File::Spec->path()) {
        my $candidate = File::Spec->catfile($directory, $program);
        return if -x $candidate && !-d $candidate;
    }
    die "Cannot find executable '$program'; install the Vulkan SDK or use --glslc PATH\n";
}

sub read_input_manifest {
    my ($directory) = @_;
    my @order;
    my %metadata;
    my $json_path = File::Spec->catfile($directory, 'library.json');
    my $index_path = File::Spec->catfile($directory, 'index.txt');

    if (-f $json_path) {
        my $decoded = eval { JSON::PP->new->decode(read_text($json_path)) };
        die "Cannot parse $json_path: $@" if $@;
        if (ref($decoded->{shaders}) eq 'ARRAY') {
            for my $entry (@{$decoded->{shaders}}) {
                my $path = ref($entry) eq 'HASH' ? $entry->{file} : $entry;
                push @order, normalize_relative_path($path)
                  if defined $path && !ref($path);
            }
        }
        %metadata = %{$decoded->{custom_uniforms}}
          if ref($decoded->{custom_uniforms}) eq 'HASH';
    } elsif (-f $index_path) {
        for my $line (split /\R/, read_text($index_path)) {
            $line =~ s/^\s+|\s+$//g;
            next if $line eq '' || $line =~ /^#/;
            push @order, normalize_relative_path($line);
        }
    }
    return (\@order, \%metadata);
}

sub normalize_relative_path {
    my ($path) = @_;
    $path =~ s{\\}{/}g;
    die "Unsafe shader path in manifest: $path\n"
      if $path =~ m{(?:^|/)\.\.(?:/|$)} || File::Spec->file_name_is_absolute($path);
    $path =~ s{^\./}{};
    return $path;
}

sub discover_shaders {
    my ($directory, $preferred, $stage) = @_;
    my @found;
    find(
        {
            no_chdir => 1,
            wanted   => sub {
                return if !-f $File::Find::name;
                if ($stage eq 'comp') {
                    return if $File::Find::name !~ /\.comp\z/i;
                } else {
                    return if $File::Find::name !~ /\.(?:glsl|frag)\z/i;
                }
                my $relative = File::Spec->abs2rel($File::Find::name, $directory);
                $relative =~ s{\\}{/}g;
                push @found, normalize_relative_path($relative);
            },
        },
        $directory
    );

    my %found = map { $_ => 1 } @found;
    my %seen;
    my @ordered;
    for my $path (@{$preferred}, sort @found) {
        next if !$found{$path} || $seen{$path}++;
        push @ordered, $path;
    }
    return @ordered;
}

sub manifest_custom_names {
    my ($metadata) = @_;
    return sort grep { /^[A-Za-z_][A-Za-z0-9_]*\z/ } keys %{$metadata};
}

sub shader_stage {
    my ($path) = @_;
    return 'comp' if $path =~ /\.comp\z/i;
    return 'frag';
}

sub output_source_name {
    my ($path, $stage) = @_;
    $path =~ s/\.(?:glsl|frag|comp)\z/.$stage/i;
    return $path;
}

sub convert_source {
    my ($source, $stage, $custom_slot) = @_;
    my @warnings;
    my %aliases;

    $source =~ s/^\x{FEFF}//;
    if ($source =~ /^\s*#version[^\r\n]*/m) {
        $source =~ s/^\s*#version[^\r\n]*/#version 450/m;
    } else {
        $source = "#version 450\n" . $source;
    }

    $source =~ s/;[ \t]+(?=uniform\s+)/;\n/g;
    $source =~ s{^\s*#define\s+USE_HISTORY_TEXTURE_ARRAY\s+0\s*$}{#define USE_HISTORY_TEXTURE_ARRAY 1}gm;

    if ($stage eq 'frag') {
        $source =~ s{^\s*(?!layout\s*\()in\s+vec2\s+tc\s*;}{layout(location = 0) in vec2 tc;}gm;
        $source =~ s{^\s*(?!layout\s*\()out\s+vec4\s+color\s*;}{layout(location = 0) out vec4 color;}gm;
    }

    $source =~ s{^\s*(?:layout\s*\([^;]*\)\s*)?uniform\s+sampler2D\s+(samp|input_image)\s*;}{'layout(set = 0, binding = 0) uniform sampler2D ' . $1 . ';'}gme;
    $source =~ s{^\s*(?:layout\s*\([^;]*\)\s*)?uniform\s+sampler2DArray\s+history\s*;}{do { push @warnings, 'uses history binding 2; ordinary ACMXVK post-process passes do not expose it yet'; 'layout(set = 0, binding = 2) uniform sampler2DArray history;' }}gme;
    $source =~ s{^\s*(?:layout\s*\([^;]*\)\s*)?uniform\s+sampler1D\s+(spectrum|spectrum0)\s*;}{'layout(set = 0, binding = 3) uniform sampler1D ' . $1 . ';'}gme;
    $source =~ s{^\s*(?:layout\s*\([^;]*\)\s*)?uniform\s+sampler1DArray\s+spectrum_history\s*;}{layout(set = 0, binding = 4) uniform sampler1DArray spectrum_history;}gm;

    if ($stage eq 'comp') {
        my $output_count = ($source =~ s{^\s*(?:layout\s*\([^;]*\)\s*)?(?:writeonly\s+)?uniform\s+image2D\s+([A-Za-z_][A-Za-z0-9_]*)\s*;}{'layout(set = 0, binding = 5, rgba8) writeonly uniform image2D ' . $1 . ';'}gme);
        push @warnings, 'compute shader has no image2D output declaration'
          if !$output_count;
    }

    $source =~ s{^\s*uniform\s+(float|int|vec2|vec4)\s+([^;]+);[^\r\n]*}{
        convert_uniform_declaration($1, $2, \%aliases, $custom_slot, \@warnings)
    }egm;

    while ($source =~ /^\s*(?:layout\s*\([^;]*\)\s*)?uniform\s+([^;]+);/gm) {
        my $declaration = $1;
        next if $declaration =~ /^(?:sampler2D\s+(?:samp|input_image)|sampler2DArray\s+history|sampler1D\s+(?:spectrum|spectrum0)|sampler1DArray\s+spectrum_history|image2D\s+)/;
        next
          if $declaration =~ /^sampler2D\s+textures\s*\[/
          && $source =~ /^\s*#define\s+USE_HISTORY_TEXTURE_ARRAY\s+1\s*$/m;
        push @warnings, "unsupported uniform declaration: $declaration";
    }

    my $abi = abi_block(\%aliases);
    $source =~ s{(^#version[^\r\n]*\r?\n(?:\s*#extension[^\r\n]*\r?\n)*)}{$1\n$abi\n}m
      or $source = "$abi\n$source";
    return ($source, \@warnings);
}

sub convert_uniform_declaration {
    my ($type, $declarations, $aliases, $custom_slot, $warnings) = @_;
    my @unhandled;
    for my $declaration (split /,/, $declarations) {
        $declaration =~ s/^\s+|\s+$//g;
        my ($name) = $declaration =~ /^([A-Za-z_][A-Za-z0-9_]*)/;
        if (!defined $name) {
            push @unhandled, $declaration;
            next;
        }
        my $alias = builtin_alias($type, $name);
        if (defined $alias) {
            $aliases->{$name} = $alias;
        } elsif ($type eq 'float' && exists $custom_slot->{$name}) {
            my $index = $custom_slot->{$name};
            my @component = qw(x y z w);
            $aliases->{$name} =
              sprintf('ext.custom_uniforms[%d].%s', int($index / 4), $component[$index % 4]);
        } else {
            push @unhandled, $declaration;
            push @{$warnings}, "unsupported $type uniform: $name";
        }
    }
    return '' if !@unhandled;
    return "uniform $type " . join(', ', @unhandled) . ';';
}

sub builtin_alias {
    my ($type, $name) = @_;
    my %float_alias = (
        alpha     => 'ext.u0.x',
        time_f    => 'ext.u2.y',
        iTime     => 'ext.u0.y',
        iTimeDelta => 'ext.u1.x',
        amp       => 'ext.u1.y',
        uamp      => 'ext.u1.y',
        iamp      => 'ext.u1.z',
        iFrameRate => 'ext.u1.w',
        iSampleRate => 'ext.u2.z',
        amp_peak  => 'ext.u2.w',
        amp_rms   => 'ext.u3.z',
        amp_smooth => 'ext.u3.w',
        amp_low   => 'ext.audio_bands.x',
        amp_mid   => 'ext.audio_bands.y',
        amp_high  => 'ext.audio_bands.z',
    );
    return $float_alias{$name} if $type eq 'float' && exists $float_alias{$name};
    return 'ext.u0.zw' if $type eq 'vec2' && $name eq 'iResolution';
    return 'ext.mouse.xy' if $type eq 'vec2' && ($name eq 'iMouse' || $name eq 'iMouseClick');
    return 'ext.mouse' if $type eq 'vec4' && $name eq 'iMouse';
    return 'int(ext.u2.x)' if $type eq 'int' && $name eq 'iFrame';
    return 'int(ext.u3.x)' if $type eq 'int' && $name eq 'history_head';
    return 'int(ext.audio_history.x)' if $type eq 'int' && $name eq 'spectrum_history_head';
    return 'int(ext.audio_history.y)' if $type eq 'int' && $name eq 'spectrum_history_size';
    return undef;
}

sub abi_block {
    my ($aliases) = @_;
    my $text = <<'GLSL';
layout(set = 0, binding = 1, std140) uniform SpriteExtended {
    vec4 mouse;
    vec4 u0;
    vec4 u1;
    vec4 u2;
    vec4 u3;
    vec4 custom_uniforms[16];
    vec4 audio_bands;
    vec4 audio_history;
} ext;
GLSL
    for my $name (sort keys %{$aliases}) {
        $text .= "#define $name $aliases->{$name}\n";
    }
    return $text;
}

sub run_compiler {
    my ($compiler, $stage, $source, $destination) = @_;
    my $error = gensym;
    my $pid = open3(undef, my $stdout, $error,
                    $compiler, "-fshader-stage=$stage", $source, '-o', $destination);
    local $/;
    my $out = <$stdout> // '';
    my $err = <$error> // '';
    waitpid($pid, 0);
    return ($? >> 8, $out . $err);
}

sub normalize_custom_metadata {
    my ($metadata) = @_;
    return {
        minimum => numeric_or($metadata->{minimum}, 0.0),
        maximum => numeric_or($metadata->{maximum}, 1.0),
        step    => numeric_or($metadata->{step}, 0.01),
        value   => numeric_or($metadata->{value}, 0.0),
    };
}

sub numeric_or {
    my ($value, $fallback) = @_;
    return $fallback if !defined $value || ref($value);
    return 0 + $value if $value =~ /^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\z/;
    return $fallback;
}

sub report_entry {
    my ($file, $status, $warnings, $details) = @_;
    my $entry = "[$status] $file";
    $entry .= "\n  warning: $_" for @{$warnings};
    if ($details ne '') {
        $details =~ s/\s+\z//;
        $entry .= "\n  $details";
    }
    return $entry;
}

sub read_text {
    my ($path) = @_;
    open my $file, '<', $path or die "Cannot read $path: $!\n";
    local $/;
    my $contents = <$file>;
    close $file or die "Cannot close $path: $!\n";
    return defined $contents ? $contents : '';
}

sub write_text {
    my ($path, $contents) = @_;
    my $temporary = "$path.tmp.$$";
    open my $file, '>', $temporary or die "Cannot write $temporary: $!\n";
    print {$file} $contents or die "Cannot write $temporary: $!\n";
    close $file or die "Cannot close $temporary: $!\n";
    rename $temporary, $path or die "Cannot replace $path: $!\n";
}
