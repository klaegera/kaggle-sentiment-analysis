#!/bin/bash

# check if arguments provided
if [ $# -lt 1 ] ; then
	echo -e "\n USAGE: $0 [text_1] ... [text_n]\n"
	exit 1
fi

# exit if an error occurs
set -e
set -o pipefail

echo -e '\n *----------------------------------\n |'
echo ' |  PROCESSING FILES'
echo -e ' |\n *----------------------------------\n'

cat "$@"             | # read input files
sed "s/ /\n/g"       | # split words into lines
grep -v "^\s*$"      | # remove empty lines
sort | uniq -c       | # count occurences
sed "s/^\s\+//g"     | # strip whitespace from uniq output
grep -v "^[1234]\s"  | # remove words occurring less than 5 times
cut -d' ' -f2        | # discard count, keeping only word
tee > vocab.txt        # save to file

echo -e '\n *----------------------------------\n |'
echo ' |  VOCAB LIST SAVED TO: vocab.txt'
echo -e ' |\n *----------------------------------\n'

