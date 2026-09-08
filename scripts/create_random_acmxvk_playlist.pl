#!/usr/bin/env perl

use strict;
use warnings;

use Getopt::Long qw(GetOptions);
use JSON::PP qw(decode_json);

use constant MAX_PLAYLIST_NODES   => 10_000;
use constant MAX_PLAYLIST_ENTRIES => 65_536;

sub usage {
    my ($exit_code) = @_;
    my $stream = $exit_code == 0 ? *STDOUT : *STDERR;
    print {$stream} <<'USAGE';
Usage:
  create_random_acmxvk_playlist.pl [options] library.json

Options:
  --entries N       Number of playlist nodes to create (1-10000)
  --max-shaders N   Maximum shaders per node; each node gets 1..N
  --output FILE     Output playlist path (default: random.playlist.txt)
  --seed N          Repeatable non-negative random seed
  --force           Replace an existing output file
  --help            Show this help

Example:
  ./scripts/create_random_acmxvk_playlist.pl \
      --entries 1000 --max-shaders 4 \
      --output vkplaylist.txt --force \
      /path/to/library.json
USAGE
    exit $exit_code;
}

my $entries;
my $max_shaders;
my $output = 'random.playlist.txt';
my $seed;
my $force = 0;
my $help = 0;

GetOptions(
    'entries|n=i'     => \$entries,
    'max-shaders|m=i' => \$max_shaders,
    'output|o=s'      => \$output,
    'seed=i'          => \$seed,
    'force'           => \$force,
    'help|h'          => \$help,
) or usage(2);

usage(0) if $help;
usage(2) if @ARGV != 1;

my $library_path = $ARGV[0];
die "--entries is required\n"
    if !defined $entries;
die "--entries must be between 1 and " . MAX_PLAYLIST_NODES . "\n"
    if $entries < 1 || $entries > MAX_PLAYLIST_NODES;
die "--max-shaders is required\n"
    if !defined $max_shaders;
die "--max-shaders must be at least 1\n"
    if $max_shaders < 1;
die "--seed must be non-negative\n"
    if defined($seed) && $seed < 0;
die "--output must not be empty\n"
    if !defined($output) || $output eq '';
die "output already exists: $output (use --force to replace it)\n"
    if -e $output && !$force;

open my $library_file, '<', $library_path
    or die "cannot open $library_path: $!\n";
local $/;
my $json_text = <$library_file>;
close $library_file
    or die "cannot close $library_path: $!\n";

my $manifest = eval { decode_json($json_text) };
die "cannot parse $library_path: $@"
    if $@;
die "$library_path must contain a JSON object\n"
    if ref($manifest) ne 'HASH';
die "$library_path must contain a shaders array\n"
    if ref($manifest->{shaders}) ne 'ARRAY';

my @shaders;
my %seen;
for my $entry (@{$manifest->{shaders}}) {
    my $name;
    if (!ref($entry)) {
        $name = $entry;
    } elsif (ref($entry) eq 'HASH' && !ref($entry->{file})) {
        $name = $entry->{file};
    }
    die "$library_path contains a shader entry without a file name\n"
        if !defined($name) || $name eq '';
    die "shader path contains a line break: $name\n"
        if $name =~ /[\r\n]/;
    die "shader path conflicts with playlist node syntax: $name\n"
        if $name =~ /^\s*\[/;

    $name =~ s{\\}{/}g;
    $name .= '.spv' if $name !~ /\.spv\z/i;
    next if $seen{$name}++;
    push @shaders, $name;
}

die "$library_path contains no usable shaders\n"
    if !@shaders;

my $effective_max = $max_shaders < @shaders ? $max_shaders : scalar @shaders;
warn "maximum shaders per node reduced to $effective_max because the library "
    . "contains only " . scalar(@shaders) . " unique shaders\n"
    if $effective_max != $max_shaders;

$seed = time ^ ($$ << 16) if !defined $seed;
srand($seed);

my @nodes;
my $total_entries = 0;
for my $node_index (1 .. $entries) {
    my $remaining_nodes = $entries - $node_index;
    my $remaining_capacity = MAX_PLAYLIST_ENTRIES - $total_entries;
    my $node_max = $remaining_capacity - $remaining_nodes;
    $node_max = $effective_max if $node_max > $effective_max;
    die "requested playlist cannot fit within ACMXVK's "
        . MAX_PLAYLIST_ENTRIES . "-shader entry limit\n"
        if $node_max < 1;

    my $shader_count = 1 + int(rand($node_max));
    my @pool = @shaders;
    my @selected;
    for my $selection_index (0 .. $shader_count - 1) {
        my $swap_index = $selection_index
            + int(rand(@pool - $selection_index));
        @pool[$selection_index, $swap_index]
            = @pool[$swap_index, $selection_index];
        push @selected, $pool[$selection_index];
    }
    push @nodes, [sprintf('Random %04d', $node_index), @selected];
    $total_entries += $shader_count;
}

my $temporary_output = "$output.tmp.$$";
open my $playlist_file, '>', $temporary_output
    or die "cannot create $temporary_output: $!\n";
print {$playlist_file} "# Random ACMXVK playlist\n";
print {$playlist_file} "# Source: $library_path\n";
print {$playlist_file} "# Seed: $seed\n";
print {$playlist_file} "# Nodes: $entries; shader entries: $total_entries; "
    . "maximum per node: $effective_max\n\n";
for my $node (@nodes) {
    my ($name, @selected) = @{$node};
    print {$playlist_file} "[$name]\n";
    print {$playlist_file} "$_\n" for @selected;
    print {$playlist_file} "\n";
}
close $playlist_file
    or die "cannot close $temporary_output: $!\n";

if ($force && -e $output) {
    unlink $output
        or die "cannot replace $output: $!\n";
}
rename $temporary_output, $output
    or die "cannot publish $output: $!\n";

print "Created $output with $entries nodes and $total_entries shader entries "
    . "(seed $seed).\n";
