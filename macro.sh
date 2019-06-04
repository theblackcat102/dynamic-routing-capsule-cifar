git checkout bs32_11
sleep 5
./run.sh 5 32
mkdir resultados_32
mv *.zip resultados_32

git add .
git commit -m 'batch size 32 11 clases'


git checkout bs16_11
sleep 5
./run.sh 5 16
mkdir resultados_16
mv *.zip resultados_16

git add .
git commit -m 'batch size 16 11 clases'