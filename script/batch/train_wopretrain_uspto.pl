#!/usr/bin/perl
use strict; 
use warnings; 
use POSIX;

my $path = "/fs/ess/PCON0041/xiaohu/MAT/script/batch/";

my $codepath = "/fs/ess/PCON0041/xiaohu/MAT/src/Model";

my $resultpath = "/fs/ess/PCON0041/xiaohu/MAT/result/USPTO";

# where to put the job files
my $job_dir = "${path}JOB20/jobs"; 
system("mkdir -p ${job_dir}") unless (-e "${job_dir}");  

# where to put the joblock files
my $joblock_dir = "${path}JOB20/joblocks";
system("mkdir -p ${joblock_dir}") unless (-e "${joblock_dir}"); 

# where to put the job output files
my $job_out = "${path}JOB20/jobout"; 
system("mkdir -p ${job_out}") unless (-e "${job_out}"); 

my @Nlayer = ("5");
my @hidden_size = ("256");
my @lrdecay = ("10", "15");
my @decaystep = ("0.85", "0.9", "0.95");
my @initlr = ("0.00001");
my @gnorm = ("1000");
my @dropout = ("0.0");
my @wd = ("0.00001", "0.000001", "0.0");
for(my $i=0;$i<=$#Nlayer;$i++){ 
    my $Nlayer = $Nlayer[$i];

    for(my $j=0;$j<=$#hidden_size;$j++){
        my $hidden_size = $hidden_size[$j];

        for(my $k=0;$k<=$#lrdecay;$k++){
            my $lrdecay = $lrdecay[$k];
            
            for(my $m=0;$m<=$#decaystep;$m++){
                my $decaystep = $decaystep[$m];

                for(my $n=0;$n<=$#initlr;$n++){
                    my $initlr = $initlr[$n];                    

                    for(my $p=0;$p<=$#gnorm;$p++){
                        my $gnorm = $gnorm[$p];

                        for(my $q=0;$q<=$#dropout;$q++){
                            my $dropout = $dropout[$q];

                            for(my $r=0;$r<=$#wd;$r++){
                                my $wd = $wd[$r];

                                # the name of the job file has the parameter configuration
                                    my $jobfile      = sprintf("%s/hs%s\_Nlayer%s\_lrdecay%s\_factor%s\_initlr%s\_gnorm%s\_dropout%s\_wd%s.job", $job_dir, $hidden_size, $Nlayer, $lrdecay, $decaystep, $initlr, $gnorm, $dropout, $wd); 	
                                    my $lockfile     = sprintf("%s/hs%s\_Nlayer%s\_lrdecay%s\_factor%s\_initlr%s\_gnorm%s\_dropout%s\_wd%s.loc", $joblock_dir, $hidden_size, $Nlayer, $lrdecay, $decaystep, $initlr, $gnorm, $dropout, $wd); 	
                                    my $outfile      = sprintf("%s/hs%s\_Nlayer%s\_lrdecay%s\_factor%s\_initlr%s\_gnorm%s\_dropout%s\_wd%s.log", $job_out, $hidden_size, $Nlayer, $lrdecay, $decaystep, $initlr, $gnorm, $dropout, $wd);
                                    my $modelpath    = sprintf("%s/hs%s\_Nlayer%s\_lrdecay%s\_factor%s\_initlr%s\_gnorm%s\_dropout%s\_wd%s", $resultpath, $hidden_size, $Nlayer, $lrdecay, $decaystep, $initlr, $gnorm, $dropout, $wd);
                                
                                    open(JOB, ">$jobfile") or die "$! $jobfile\n"; 
                                
                                    print JOB "nohup "; 
                                    print JOB "python "; # where matlab lis
                                    printf JOB "%s/mainUSPTOpretrain_new.py ", $codepath;
                                    printf JOB "--epochs 35 --batch_size 32 --split 1 --use_pretrain 0 --molecule_pooling_method 'con' --self_hidden_dim %d --self_n_layers %d --patience %d --factor %f --init_lr %f --gnorm %f --dropout %f --weight_decay %f --out_dir %s > %s 2>&1 ", $hidden_size, $Nlayer, $lrdecay, $decaystep, $initlr, $gnorm, $dropout, $wd, $modelpath, $outfile;  # your matlab script 
                                    print JOB "\n";
                                        
                                    close(JOB);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
