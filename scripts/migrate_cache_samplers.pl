#!/usr/bin/env perl

use strict;
use warnings;

use File::Copy qw(copy);
use File::Find qw(find);
use File::Spec;
use Getopt::Long qw(GetOptions);

my $dry_run = 0;
my $backup = 0;
my $help = 0;

GetOptions(
    'dry-run' => \$dry_run,
    'backup'  => \$backup,
    'help'    => \$help,
) or usage(1);

usage(0) if $help;

my $shader_dir = shift @ARGV // 'shaders';
usage(1) if @ARGV;
die "Shader directory does not exist: $shader_dir\n" if !-d $shader_dir;

my @shader_files;
find(
    {
        no_chdir => 1,
        wanted   => sub {
            return if !-f $_;
            return if $_ !~ /\.glsl\z/i;
            push @shader_files, $File::Find::name;
        },
    },
    $shader_dir
);

my $changed_files = 0;
my $declaration_count = 0;
my $lookup_count = 0;

for my $path (sort @shader_files) {
    open my $input, '<', $path or die "Cannot read $path: $!\n";
    local $/;
    my $original = <$input>;
    close $input or die "Cannot close $path: $!\n";

    my ($migrated, $declarations, $lookups) = migrate_shader($original);
    next if $migrated eq $original;

    ++$changed_files;
    $declaration_count += $declarations;
    $lookup_count += $lookups;

    if ($dry_run) {
        print "would migrate $path ($declarations declarations, $lookups lookups)\n";
        next;
    }

    if ($backup) {
        my $backup_path = "$path.bak";
        copy($path, $backup_path)
          or die "Cannot create backup $backup_path: $!\n";
    }

    my ($volume, $directories, $filename) = File::Spec->splitpath($path);
    my $temporary =
      File::Spec->catpath($volume, $directories, ".$filename.tmp.$$");
    open my $output, '>', $temporary
      or die "Cannot write $temporary: $!\n";
    print {$output} $migrated
      or die "Cannot write shader contents to $temporary: $!\n";
    close $output or die "Cannot close $temporary: $!\n";
    rename $temporary, $path
      or die "Cannot replace $path with $temporary: $!\n";

    print "migrated $path ($declarations declarations, $lookups lookups)\n";
}

my $verb = $dry_run ? 'would migrate' : 'migrated';
print "$verb $changed_files shader files; replaced "
  . "$declaration_count declarations and $lookup_count lookups\n";

sub migrate_shader {
    my ($source) = @_;

    return ($source, 0, 0)
      if $source !~ /\buniform\s+sampler2D[^;]*\bsamp[1-8]\b[^;]*;/
      && $source !~ /\buniform\s+sampler2D\s+textures\s*\[/
      && $source !~ /\btexture(?:2D)?\s*\(\s*samp[1-8]\s*,/
      && $source !~ /\btexture(?:2D)?\s*\(\s*textures\s*\[/;

    my $has_history =
      $source =~ /\buniform\s+sampler2DArray\s+history\s*;/;
    my $inserted_history = $has_history ? 1 : 0;
    my $declarations = 0;

    $source =~ s{
        ^([ \t]*)uniform[ \t]+sampler2D[ \t]+
        ([^;\r\n]*\bsamp[1-8]\b[^;\r\n]*)
        ;([^\r\n]*)(\r?\n|\z)
    }{
        my ($indent, $names, $suffix, $newline) = ($1, $2, $3, $4);
        my @names = map {
            my $name = $_;
            $name =~ s/^\s+|\s+$//g;
            $name;
        } split /,/, $names;
        my @kept;
        for my $name (@names) {
            if ($name =~ /\Asamp[1-8]\z/) {
                ++$declarations;
            } else {
                push @kept, $name;
            }
        }
        my $replacement = "";
        if (@kept) {
            $replacement =
              $indent . "uniform sampler2D " . join(", ", @kept) . ";"
              . $suffix . $newline;
        }
        if (!$inserted_history) {
            $inserted_history = 1;
            $replacement .= history_block($indent, $newline);
        }
        $replacement;
    }egmx;

    $source =~ s{
        ^([ \t]*)uniform[ \t]+sampler2D[ \t]+textures[ \t]*
        \[[^\]\r\n]+\][ \t]*;[^\r\n]*(\r?\n|\z)
    }{
        ++$declarations;
        my $replacement = "";
        if (!$inserted_history) {
            $inserted_history = 1;
            $replacement = history_block($1, $2);
        }
        $replacement;
    }egmx;

    my ($migrated, $lookups) = migrate_texture_calls($source);
    if ($lookups > 0 && !$inserted_history) {
        my $declaration = history_block('', "\n");
        if ($migrated =~ /\A([^\r\n]*\#version[^\r\n]*\r?\n)/) {
            substr($migrated, length($1), 0, $declaration);
        } else {
            $migrated = $declaration . $migrated;
        }
    }
    return ($migrated, $declarations, $lookups);
}

sub migrate_texture_calls {
    my ($source) = @_;

    my $output = '';
    my $position = 0;
    my $length = length $source;
    my $lookups = 0;
    my $state = 'code';

    while ($position < $length) {
        if ($state eq 'line-comment') {
            my $character = substr($source, $position, 1);
            $output .= $character;
            ++$position;
            $state = 'code' if $character eq "\n";
            next;
        }

        if ($state eq 'block-comment') {
            if (substr($source, $position, 2) eq '*/') {
                $output .= '*/';
                $position += 2;
                $state = 'code';
            } else {
                $output .= substr($source, $position, 1);
                ++$position;
            }
            next;
        }

        if (substr($source, $position, 2) eq '//') {
            $output .= '//';
            $position += 2;
            $state = 'line-comment';
            next;
        }
        if (substr($source, $position, 2) eq '/*') {
            $output .= '/*';
            $position += 2;
            $state = 'block-comment';
            next;
        }

        my $remaining = substr($source, $position);
        if ($remaining =~
            /\Atexture(?:2D)?([ \t\r\n]*)\(([ \t\r\n]*)textures[ \t\r\n]*\[[ \t\r\n]*([^\]\r\n]+?)[ \t\r\n]*\]([ \t\r\n]*),/)
        {
            my $spacing_after_name = $1;
            my $spacing_after_open = $2;
            my $index = $3;
            my $spacing_before_comma = $4;
            my $matched_length = length $&;
            my $coordinate_start = $position + $matched_length;
            my ($coordinate_end, $delimiter) =
              find_coordinate_end($source, $coordinate_start);

            if (defined $coordinate_end) {
                my $coordinate =
                  substr($source, $coordinate_start,
                         $coordinate_end - $coordinate_start);
                $coordinate =~ s/^\s+|\s+$//g;
                $index =~ s/^\s+|\s+$//g;
                $output .=
                    'texture'
                  . $spacing_after_name
                  . '('
                  . $spacing_after_open
                  . 'history'
                  . $spacing_before_comma
                  . ', vec3('
                  . $coordinate
                  . ", float(CACHE_HISTORY_LAYER($index)))";
                $position = $coordinate_end;
                ++$lookups;
                next;
            }
        }

        if ($remaining =~
            /\Atexture(?:2D)?([ \t\r\n]*)\(([ \t\r\n]*)samp([1-8])([ \t\r\n]*),/)
        {
            my $spacing_after_name = $1;
            my $spacing_after_open = $2;
            my $sampler_number = $3;
            my $spacing_before_comma = $4;
            my $matched_length = length $&;
            my $coordinate_start = $position + $matched_length;
            my ($coordinate_end, $delimiter) =
              find_coordinate_end($source, $coordinate_start);

            if (defined $coordinate_end) {
                my $coordinate =
                  substr($source, $coordinate_start,
                         $coordinate_end - $coordinate_start);
                $coordinate =~ s/^\s+|\s+$//g;
                my $layer = $sampler_number - 1;
                $output .=
                    'texture'
                  . $spacing_after_name
                  . '('
                  . $spacing_after_open
                  . 'history'
                  . $spacing_before_comma
                  . ', vec3('
                  . $coordinate
                  . ", float(CACHE_HISTORY_LAYER($layer)))";
                $position = $coordinate_end;
                ++$lookups;
                next;
            }
        }

        $output .= substr($source, $position, 1);
        ++$position;
    }

    return ($output, $lookups);
}

sub find_coordinate_end {
    my ($source, $start) = @_;

    my $round_depth = 0;
    my $square_depth = 0;
    my $brace_depth = 0;
    my $position = $start;
    my $length = length $source;
    my $state = 'code';

    while ($position < $length) {
        my $pair = substr($source, $position, 2);
        my $character = substr($source, $position, 1);

        if ($state eq 'line-comment') {
            $state = 'code' if $character eq "\n";
            ++$position;
            next;
        }
        if ($state eq 'block-comment') {
            if ($pair eq '*/') {
                $position += 2;
                $state = 'code';
            } else {
                ++$position;
            }
            next;
        }
        if ($pair eq '//') {
            $position += 2;
            $state = 'line-comment';
            next;
        }
        if ($pair eq '/*') {
            $position += 2;
            $state = 'block-comment';
            next;
        }

        if ($character eq '(') {
            ++$round_depth;
        } elsif ($character eq ')') {
            return ($position, ')')
              if $round_depth == 0
              && $square_depth == 0
              && $brace_depth == 0;
            --$round_depth;
        } elsif ($character eq '[') {
            ++$square_depth;
        } elsif ($character eq ']') {
            --$square_depth;
        } elsif ($character eq '{') {
            ++$brace_depth;
        } elsif ($character eq '}') {
            --$brace_depth;
        } elsif (
            $character eq ','
            && $round_depth == 0
            && $square_depth == 0
            && $brace_depth == 0
          )
        {
            return ($position, ',');
        }

        ++$position;
    }

    return;
}

sub history_block {
    my ($indent, $newline) = @_;
    return
        $indent . "uniform sampler2DArray history;" . $newline
      . $indent . "uniform int history_head;" . $newline
      . "#ifndef SIZE" . $newline
      . "#define SIZE 8" . $newline
      . "#endif" . $newline
      . "#ifndef CACHE_HISTORY_LAYER" . $newline
      . "#define CACHE_HISTORY_LAYER(index) "
      . "((history_head + (index)) % SIZE)" . $newline
      . "#endif" . $newline;
}

sub usage {
    my ($exit_code) = @_;
    print <<'USAGE';
Usage: scripts/migrate_cache_samplers.pl [options] [shader-directory]

Replace legacy cache uniforms samp1 through samp8 and textures[SIZE] with one
sampler2DArray named history. Calls such as:

    texture(samp3, uv)

become:

    texture(history, vec3(uv, float(CACHE_HISTORY_LAYER(2))))

The default shader directory is ./shaders.

Options:
    --dry-run   Report files that would change without writing them.
    --backup    Save each original shader beside it with a .bak suffix.
    --help      Show this help.
USAGE
    exit $exit_code;
}
