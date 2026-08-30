#!/usr/bin/env perl

use strict;
use warnings;

use File::Basename qw(dirname);
use File::Find qw(find);
use File::Spec;
use File::Temp qw(tempfile);
use Getopt::Long qw(GetOptions);
use JSON::PP qw(decode_json);

my $root = '';
my $output = '';
my $help = 0;

GetOptions(
    'root=s'   => \$root,
    'output=s' => \$output,
    'help'     => \$help,
) or usage(1);
usage(0) if $help;
die "--root is required\n" if $root eq '';

$root = File::Spec->rel2abs($root);
die "Shader source directory does not exist: $root\n" if !-d $root;
$output = File::Spec->catfile($root, 'library.json') if $output eq '';
$output = File::Spec->rel2abs($output);

my @fragment_files;
my @compute_files;
# Keep the converter's ABI stable even after --prune removes the last shader
# that happens to reference a slot. Later slots cannot be renumbered because
# the compiled GLSL aliases address these positions directly.
my %slot_names = (
    0  => 'square_size',
    1  => 'alpha_value',
    2  => 'alpha_r',
    3  => 'alpha_g',
    4  => 'alpha_b',
    5  => 'value_alpha_r',
    6  => 'value_alpha_g',
    7  => 'value_alpha_b',
    8  => 'index_value',
    9  => 'restore_black',
    10 => 'seed',
    11 => 'value1',
    12 => 'iChannelTime',
    13 => 'time_speed',
    14 => 'blendAmt',
    15 => 'uDistortion',
    16 => 'uRotateSpeed',
    17 => 'uWarpSpeed',
    18 => 'uRandRate',
    19 => 'uPhaseRate',
    20 => 'slider1',
    21 => 'slider2',
    22 => 'slider3',
    23 => 'slider4',
    24 => 'frequency',
    25 => 'strength',
    26 => 'random_seed',
);
my %name_slots = reverse %slot_names;

find(
    {
        no_chdir => 1,
        wanted   => sub {
            return if !-f $_;
            my $path = $File::Find::name;
            my $relative = File::Spec->abs2rel($path, $root);
            $relative =~ s{\\}{/}g;
            if ($relative =~ m{(?:^|/)compute/} && $relative =~ /\.comp\z/) {
                push @compute_files, $relative;
            } elsif ($relative !~ m{(?:^|/)compute/} &&
                     $relative =~ /\.frag\z/) {
                push @fragment_files, $relative;
            } else {
                return;
            }
            scan_custom_slots($path, \%slot_names, \%name_slots);
        },
    },
    $root
);

@fragment_files = sort { lc($a) cmp lc($b) || $a cmp $b } @fragment_files;
@compute_files = sort { lc($a) cmp lc($b) || $a cmp $b } @compute_files;
my @shaders = (@fragment_files, @compute_files);
die "No .frag or compute/*.comp sources found in $root\n" if !@shaders;
die "Shader library exceeds ACMXVK's 16384-entry limit\n"
  if @shaders > 16_384;

my %output_names;
for my $shader (@shaders) {
    my $compiled = lc("$shader.spv");
    die "Case-insensitive duplicate shader output: $shader.spv\n"
      if $output_names{$compiled}++;
}

my $maximum_slot = 0;
for my $slot (keys %slot_names) {
    $maximum_slot = $slot if $slot > $maximum_slot;
}
for my $slot (0 .. $maximum_slot) {
    die "No custom-uniform name was found for slot $slot\n"
      if !exists $slot_names{$slot};
}

my %metadata = (
    square_size => [1.0, 128.0, 1.0, 55.0],
    slider1     => [0.0, 1.0, 0.01, 0.5],
    slider2     => [0.0, 1.0, 0.01, 0.6],
    slider3     => [0.0, 1.0, 0.01, 0.35],
    slider4     => [0.0, 1.0, 0.01, 0.8],
);

# Preserve edited ranges and values when repairing or refreshing an existing
# source manifest. Slots are always regenerated from the shader ABI discovered
# above, because stale/missing slots are exactly what this tool must repair.
if (-f $output) {
    open my $existing_file, '<', $output
      or die "Cannot read existing manifest $output: $!\n";
    local $/;
    my $existing_text = <$existing_file>;
    close $existing_file;
    my $existing = eval { decode_json($existing_text) };
    die "Cannot parse existing manifest $output: $@\n" if $@;
    if (ref($existing->{custom_uniforms}) eq 'HASH') {
        for my $name (keys %{ $existing->{custom_uniforms} }) {
            my $uniform = $existing->{custom_uniforms}{$name};
            next if ref($uniform) ne 'HASH';
            my $fallback = $metadata{$name} // [0.0, 1.0, 0.01, 0.0];
            $metadata{$name} = [
                $uniform->{minimum} // $fallback->[0],
                $uniform->{maximum} // $fallback->[1],
                $uniform->{step}    // $fallback->[2],
                $uniform->{value}   // $fallback->[3],
            ];
        }
    }
}

my ($temporary, $temporary_path) = tempfile(
    'library.json.tmp.XXXXXX',
    DIR    => dirname($output),
    UNLINK => 0,
);

print {$temporary} "{\n    \"version\": 1,\n";
print {$temporary} "    \"backend\": \"acmxvk\",\n";
print {$temporary} "    \"library_type\": \"source\",\n";
print {$temporary} "    \"custom_uniforms\": {\n";
for my $slot (0 .. $maximum_slot) {
    my $name = $slot_names{$slot};
    my ($minimum, $maximum, $step, $value) =
      @{ $metadata{$name} // [0.0, 1.0, 0.01, 0.0] };
    print {$temporary} '        "', json_escape($name), "\": {\n";
    print {$temporary} "            \"slot\": $slot,\n";
    print {$temporary} "            \"minimum\": $minimum,\n";
    print {$temporary} "            \"maximum\": $maximum,\n";
    print {$temporary} "            \"step\": $step,\n";
    print {$temporary} "            \"value\": $value\n";
    print {$temporary} '        }',
      ($slot < $maximum_slot ? ",\n" : "\n");
}
print {$temporary} "    },\n    \"shaders\": [\n";
for my $index (0 .. $#shaders) {
    print {$temporary} '        "', json_escape($shaders[$index]), '"',
      ($index < $#shaders ? ",\n" : "\n");
}
print {$temporary} "    ]\n}\n";
close $temporary or die "Cannot close $temporary_path: $!\n";
rename $temporary_path, $output
  or die "Cannot replace $output: $!\n";

print "Wrote $output\n";
print scalar(@fragment_files), " fragment shader(s), ",
  scalar(@compute_files), " compute shader(s), ",
  ($maximum_slot + 1), " custom-uniform slot(s)\n";

sub scan_custom_slots {
    my ($path, $slots, $names) = @_;
    open my $input, '<', $path or die "Cannot read $path: $!\n";
    while (my $line = <$input>) {
        next
          if $line !~ /^\s*\#define\s+([A-Za-z_][A-Za-z0-9_]*)\s+
                         ext\.custom_uniforms\[(\d+)\]\.([xyzw])/x;
        my ($name, $vector, $component) = ($1, $2, $3);
        my %component_index = (x => 0, y => 1, z => 2, w => 3);
        my $slot = $vector * 4 + $component_index{$component};
        if (exists $slots->{$slot} && $slots->{$slot} ne $name) {
            die "Custom-uniform slot $slot is both $slots->{$slot} and "
              . "$name ($path)\n";
        }
        if (exists $names->{$name} && $names->{$name} != $slot) {
            die "Custom uniform $name uses slots $names->{$name} and "
              . "$slot ($path)\n";
        }
        $slots->{$slot} = $name;
        $names->{$name} = $slot;
    }
    close $input;
}

sub json_escape {
    my ($value) = @_;
    $value =~ s/\\/\\\\/g;
    $value =~ s/"/\\"/g;
    $value =~ s/\x08/\\b/g;
    $value =~ s/\x0c/\\f/g;
    $value =~ s/\n/\\n/g;
    $value =~ s/\r/\\r/g;
    $value =~ s/\t/\\t/g;
    return $value;
}

sub usage {
    my ($status) = @_;
    print <<'USAGE';
Create an ACMXVK source library.json from converted Vulkan GLSL sources.

Usage:
  create_acmxvk_source_manifest.pl --root DIR [--output FILE]

The root is scanned recursively for fragment .frag files outside compute/
and compute .comp files beneath compute/. Custom-uniform slots are inferred
from converted #define aliases. Slot zero is the legacy square_size entry.
USAGE
    exit $status;
}
