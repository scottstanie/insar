#!/bin/bash

# USAGE: run in main sentinel folder, where extra_files/ dir is and all symlinks are
rm -f ./*
mv extra_files/* .
rmdir extra_files
