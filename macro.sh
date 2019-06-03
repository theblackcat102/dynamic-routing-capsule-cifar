./run.sh 5 128
mkdir resultados_128
mv *.zip resultados_128

./run.sh 5 32
mkdir resultados_32
mv *.zip resultados_32

./run.sh 5 16
mkdir resultados_16
mv *.zip resultados_16