rm -rvf results*
rm -rvf weightsKTH/*.h5

python3 main.py --epocs 100 --dataset 5
zip -r resultsKTH_relu.zip resultsKTH/
rm -rvf resultsKTH/
rm -rvf weightsKTH/*.h5


python3 main.py --epocs 100 --dataset 5 --is_relu 0
zip -r resultsKTH_leaky_relu.zip resultsKTH/
rm -rvf resultsKTH/
rm -rvf weightsKTH/*.h5


python3 main.py --epocs 100 --dataset 5 --has 0
zip -r resultsKTH_no_act.zip resultsKTH/
rm -rvf resultsKTH/
rm -rvf weightsKTH/*.h5


python3 main.py --epocs 100 --dataset 5 --normalize 1
zip -r resultsKTH_relu_norm.zip resultsKTH/
rm -rvf resultsKTH/
rm -rvf weightsKTH/*.h5


python3 main.py --epocs 100 --dataset 5 --is_relu 0 --normalize 1
zip -r resultsKTH_leaky_relu_norm.zip resultsKTH/
rm -rvf resultsKTH/
rm -rvf weightsKTH/*.h5


python3 main.py --epocs 100 --dataset 5 --has 0 --normalize 1
zip -r resultsKTH_no_act_norm.zip resultsKTH/
rm -rvf resultsKTH/
rm -rvf weightsKTH/*.h5