#!/usr/bin/env perl
use strict;
use warnings;

if (@ARGV != 1) {
    print "Usage: ./format_shaders.pl \"file1,file2,file3\"\n";
    exit(1);
}

my $input_string = $ARGV[0];
my @filenames = split(',', $input_string);

my $formatted_output = "";

foreach my $file (@filenames) {
    # Strip any accidental whitespace around the filename
    $file =~ s/^\s+|\s+$//g;
    
    my $len = length($file);
    $formatted_output .= "${len}:${file}";
}

print "'--shader-pass-files' '$formatted_output'\n";
