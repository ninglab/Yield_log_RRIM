#!/usr/bin/perl
#
# pbsdrone <in-directory> <out-directory>
#
# This file runs psi-blast on all the files that reside in the
# in-directory and writes the results in the current directory.
#
# The routine considers only the *.fa files
#
# 
use Fcntl qw(:DEFAULT :flock);

printf("working on machine "); 
system('cat /etc/hostname'); 
$indirname  = $ARGV[0];
$outdirname = $ARGV[1];

die "Usage: pbsDrone <in-directory> <out-directory>\n" unless ($#ARGV >= 1);

-e $indirname or die "***Error: Input directory ", $indirname, " does not exist.\n";
-e $outdirname or die "***Error: Output directory ", $outdirname, " does not exist.\n";

print "Running pbsDrone.pl ...\n"; 
#==============================================================================
# Get the list of filenames from the directory that end in .job
#==============================================================================
opendir(DIR, $indirname);
@files = grep {/\.job/} readdir(DIR);
closedir(DIR);


#==============================================================================
# Perform PSI-Blast for each sequence in the files
#==============================================================================
#foreach $filename (reverse(@files)) {
$found = -1;
foreach $filename (@files) {
  $infile   = $indirname . "/" . $filename;
  $outfile  = $outdirname . "/" . $filename . ".out";

  printf "file1 %s \n" , $filename;
  next if -e $outfile;

  printf "file3 %s \n" , $filename;
# the following is to quickly create a "lock" file
  $lockfile = $outfile . ".lock";
  next if -e $lockfile;
  
  printf "file2 %s \n" , $filename;

  sysopen(LOCKFH, $lockfile, O_WRONLY | O_CREAT) or die "can't open filename $!";;
  if (!flock(LOCKFH, LOCK_EX | LOCK_NB)) { 
    # Someone beat us to creating this file. Go to next sequence.
    print "go to next 1 \n";
    close(LOCKFH);
    next;
  }

  if (-e $outfile) {
    # By the time we got the lock, someone beat us to creating the profile. Go to next sequence.
    print "go to next 2 \n";
    close(LOCKFH);
    unlink($lockfile);
    next;
  }

  #-----------------------------------------------------------------------------------
  print "Working on job $filename stored at $infile and generating $outfile.\n";

  open(FPIN, "<$infile");
  $cmd = <FPIN>;
  chomp($cmd);
  close(FPIN);

  print "Executing... $cmd\n";
  $output = `$cmd`;

  open(FPOUT, ">$outfile");
  print FPOUT $output, "\n";
  close(FPOUT);
  #-----------------------------------------------------------------------------------


  close(LOCKFH);
  unlink($lockfile);
}
#system("cat /project/dminers18/project/mRec/trunk/jobs/email.txt | sendmail -oi -t"); 
#printf "Sleeping offff\n";
#sleep;

