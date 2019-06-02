clear
ls -alh
echo borrar todo
rm -rvf results*
rm -rvf resultsKTH/
rm -rvf weightsKTH/*.h5
echo simple train
clear
ls -alh
echo begin
python3 main.py --epocs 100 --dataset $1 --lear_rate 0.01 --batch_size $2
zip -r resultsKTH_relu.zip resultsKTH/
rm -rvf resultsKTH/
rm -rvf weightsKTH/*.h5


#python3 main.py --epocs 100 --dataset $1 --is_relu 0 --lear_rate 0.05 --batch_size 32
#zip -r resultsKTH_leaky_relu.zip resultsKTH/
#rm -rvf resultsKTH/
#rm -rvf weightsKTH/*.h5


python3 main.py --epocs 100 --dataset $1 --has 0 --lear_rate 0.01 --batch_size $2
zip -r resultsKTH_no_act.zip resultsKTH/
rm -rvf resultsKTH/
rm -rvf weightsKTH/*.h5

echo normalizado

python3 main.py --epocs 100 --dataset $1 --version _norm --lear_rate 0.01 --batch_size $2
zip -r resultsKTH_relu_norm.zip resultsKTH/
rm -rvf resultsKTH/
rm -rvf weightsKTH/*.h5


#python3 main.py --epocs 100 --dataset $1 --is_relu 0 --version _norm --lear_rate 0.05 --batch_size 32
#zip -r resultsKTH_leaky_relu_norm.zip resultsKTH/
#rm -rvf resultsKTH/
#rm -rvf weightsKTH/*.h5


python3 main.py --epocs 100 --dataset $1 --has 0 --version _norm --lear_rate 0.01 --batch_size $2
zip -r resultsKTH_no_act_norm.zip resultsKTH/
rm -rvf resultsKTH/
rm -rvf weightsKTH/*.h5


echo sample particion

python3 main.py --epocs 100 --dataset $1 --version _sample --lear_rate 0.01 --batch_size $2
zip -r resultsKTH_relu_sample.zip resultsKTH/
rm -rvf resultsKTH/
rm -rvf weightsKTH/*.h5


#python3 main.py --epocs 100 --dataset $1 --is_relu 0 --version _sample --lear_rate 0.05 --batch_size 32
#zip -r resultsKTH_leaky_relu_sample.zip resultsKTH/
#rm -rvf resultsKTH/
#rm -rvf weightsKTH/*.h5


python3 main.py --epocs 100 --dataset $1 --has 0 --version _sample --lear_rate 0.01 --batch_size $2
zip -r resultsKTH_no_act_sample.zip resultsKTH/
rm -rvf resultsKTH/
rm -rvf weightsKTH/*.h5


echo sample norm

python3 main.py --epocs 100 --dataset $1 --version _sample_norm --lear_rate 0.01 --batch_size $2
zip -r resultsKTH_relu_sample_norm.zip resultsKTH/
rm -rvf resultsKTH/
rm -rvf weightsKTH/*.h5


#python3 main.py --epocs 100 --dataset $1 --is_relu 0 --version _sample_norm --lear_rate 0.05 --batch_size 32
#zip -r resultsKTH_leaky_relu_sample_norm.zip resultsKTH/
#rm -rvf resultsKTH/
#rm -rvf weightsKTH/*.h5


python3 main.py --epocs 100 --dataset $1 --has 0 --version _sample_norm --lear_rate 0.01 --batch_size $2
zip -r resultsKTH_no_act_sample_norm.zip resultsKTH/
rm -rvf resultsKTH/
rm -rvf weightsKTH/*.h5
