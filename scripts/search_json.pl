#!/usr/bin/env perl
use strict;
use warnings;
use JSON;

if (@ARGV < 2) {
    print "Usage: ./search_library.pl <library.json> <search_string>\n";
    exit(1);
}

my ($filename, $search_string) = @ARGV;

open(my $fh, '<', $filename) or die "Could not open $filename: $!\n";
local $/; 
my $json_text = <$fh>;
close($fh);

my $data = decode_json($json_text);

my $array_ref;
if (ref($data) eq 'ARRAY') {
    $array_ref = $data;
} elsif (ref($data) eq 'HASH') {
    $array_ref = $data->{'library'} || $data->{'shaders'} || $data->{'items'};
}

unless ($array_ref && ref($array_ref) eq 'ARRAY') {
    die "Error: Could not find an array to search in the JSON structure.\n";
}

for (my $i = 0; $i < @$array_ref; $i++) {
    my $item = $array_ref->[$i];
    my $string_to_search = ref($item) ? encode_json($item) : $item;
    
    if (index($string_to_search, $search_string) >= 0) {
        print $string_to_search . ":" .  $i . "\n";
    }
}
