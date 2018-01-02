# An Open Deep Learning Benchmark

Preview at http://dlmark.org.s3-website-us-west-2.amazonaws.com/

## How to use

1. Open an Ubuntu 16.04 instance on EC2
1. Git clone this repo
1. Install drivers `bash scripts/ubuntu16-cuda90-conda.sh` 
1. Run a benchmark `./benchmark gluon/cnn.py`
1. Publish results `bash build/build.sh`	
