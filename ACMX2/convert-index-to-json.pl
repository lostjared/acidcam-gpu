#!/usr/bin/env perl

use strict;
use warnings;

use File::Basename qw(dirname);
use File::Spec;
use File::Temp qw(tempfile);
use Getopt::Long qw(GetOptions);
use JSON::PP;

sub usage {
    my ($exit_code) = @_;
    my $stream = $exit_code == 0 ? *STDOUT : *STDERR;
    print {$stream} <<'USAGE';
Usage: convert-index-to-json.pl [options] <index.txt|library-directory>

Convert a legacy shader index.txt into library.json. Shader order and duplicate
entries are preserved; leading/trailing whitespace and blank lines are removed.

Options:
  -o, --output <file>  Write to this path instead of sibling library.json
  -f, --force          Overwrite an existing output file
  -h, --help           Show this help

Examples:
  ./convert-index-to-json.pl ./shaders
  ./convert-index-to-json.pl ./shaders/index.txt
  ./convert-index-to-json.pl --force ./shaders
USAGE
    exit $exit_code;
}

my $output_path;
my $force = 0;
my $help = 0;
GetOptions(
    'output|o=s' => \$output_path,
    'force|f'    => \$force,
    'help|h'     => \$help,
) or usage(2);

usage(0) if $help;
usage(2) unless @ARGV == 1;

my $input_path = $ARGV[0];
if (-d $input_path) {
    $input_path = File::Spec->catfile($input_path, 'index.txt');
}

die "Input index not found: $input_path\n" unless -f $input_path;

if (!defined $output_path) {
    $output_path = File::Spec->catfile(dirname($input_path), 'library.json');
}

die "Input and output paths must be different\n"
    if File::Spec->rel2abs($input_path) eq File::Spec->rel2abs($output_path);

if (-e $output_path && !$force) {
    die "Output already exists: $output_path (use --force to overwrite it)\n";
}

open my $input, '<:encoding(UTF-8)', $input_path
    or die "Could not open $input_path: $!\n";

my @shaders;
while (my $line = <$input>) {
    $line =~ s/^\s+//;
    $line =~ s/\s+$//;
    push @shaders, $line if length $line;
}
close $input or die "Could not close $input_path: $!\n";

my $json = JSON::PP->new->utf8->pretty->canonical->encode(
    {
        version => 1,
        shaders => \@shaders,
    }
);

my $output_directory = dirname($output_path);
die "Output directory not found: $output_directory\n"
    unless -d $output_directory;

my ($temporary, $temporary_path) = tempfile(
    '.library.json.XXXXXX',
    DIR    => $output_directory,
    UNLINK => 1,
);
binmode $temporary, ':raw'
    or die "Could not configure temporary output $temporary_path: $!\n";
print {$temporary} $json
    or die "Could not write temporary output $temporary_path: $!\n";
close $temporary
    or die "Could not finish temporary output $temporary_path: $!\n";

my $current_umask = umask();
umask($current_umask);
my $output_permissions = 0666 & ~$current_umask;
chmod $output_permissions, $temporary_path
    or die "Could not set permissions on $temporary_path: $!\n";

if ($force && -e $output_path) {
    unlink $output_path
        or die "Could not replace $output_path: $!\n";
}
rename $temporary_path, $output_path
    or die "Could not move $temporary_path to $output_path: $!\n";

print "Wrote $output_path with " . scalar(@shaders) . " shader entries.\n";
