#!/usr/bin/env perl
use strict;
use warnings;
use JSON;

if (@ARGV < 1) {
    print "Usage: ./dump_library.pl <library.json> [offset]\n";
    exit(1);
}

my ($filename, $offset) = @ARGV;
$offset //= 0; 

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
    die "Error: Could not find an array in the JSON structure.\n";
}

my @sorted_items = sort {
    my $val_a = ref($a) eq 'HASH' ? ($a->{'name'} || $a->{'file'} || encode_json($a)) : $a;
    my $val_b = ref($b) eq 'HASH' ? ($b->{'name'} || $b->{'file'} || encode_json($b)) : $b;
    lc($val_a) cmp lc($val_b);
} @$array_ref;

for (my $i = 0; $i < @sorted_items; $i++) {
    my $item = $sorted_items[$i];
    
    my $stringified = ref($item) ? encode_json($item) : $item;
    
    my $final_index = $i + $offset;
    print "$final_index: $stringified\n";
}
