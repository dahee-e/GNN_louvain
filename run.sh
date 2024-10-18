python gnn_louvain.py --network ./dataset/strike
python gnn_louvain.py --network ./dataset/football
python gnn_louvain.py --network ./dataset/karate
python gnn_louvain.py --network ./dataset/polbooks
python gnn_louvain.py --network ./dataset/polblogs
python gnn_louvain.py --network ./dataset/mexican
python gnn_louvain.py --network ./dataset/dolphin
python gnn_louvain.py --network ./dataset/railway
python gnn_louvain.py --network ./dataset/citeseer
python gnn_louvain.py --network ./dataset/cora
python gnn_louvain.py --network ./dataset/pubmed

python gnn_louvain.py --network ./dataset/karate --num_trials 10 --dropout 0.1 > ./output/karate_dropout_0.1.txt &&
python gnn_louvain.py --network ./dataset/karate --num_trials 10 --dropout 0.2 > ./output/karate_dropout_0.2.txt &&
python gnn_louvain.py --network ./dataset/karate --num_trials 10 --dropout 0.3 > ./output/karate_dropout_0.3.txt &&
python gnn_louvain.py --network ./dataset/karate --num_trials 10 --dropout 0.4 > ./output/karate_dropout_0.4.txt &&
python gnn_louvain.py --network ./dataset/karate --num_trials 10 --dropout 0.2 > ./output/karate_dropout_0.2.txt &&
python gnn_louvain.py --network ./dataset/karate --num_trials 10 --dropout 0.6 > ./output/karate_dropout_0.6.txt &&
python gnn_louvain.py --network ./dataset/karate --num_trials 10 --dropout 0.7 > ./output/karate_dropout_0.7.txt &&
python gnn_louvain.py --network ./dataset/karate --num_trials 10 --dropout 0.8 > ./output/karate_dropout_0.8.txt &&
python gnn_louvain.py --network ./dataset/karate --num_trials 10 --dropout 0.9 > ./output/karate_dropout_0.9.txt &&
python gnn_louvain.py --network ./dataset/karate --num_trials 10 --dropout 1.0 > ./output/karate_dropout_1.0.txt &


python gnn_louvain.py --network ./dataset/karate --num_trials 10 --dropout 0.2 > ./output/karate_dropout_0.2_new.txt &&
python gnn_louvain.py --network ./dataset/strike --num_trials 10 --dropout 0.2 > ./output/strike_dropout_0.2_new.txt &&
python gnn_louvain.py --network ./dataset/football --num_trials 10 --dropout 0.2 > ./output/football_dropout_0.2_new.txt &&
python gnn_louvain.py --network ./dataset/polbooks --num_trials 10 --dropout 0.2 > ./output/polbooks_dropout_0.2_new.txt &&
python gnn_louvain.py --network ./dataset/polblogs --num_trials 10 --dropout 0.2 > ./output/polblogs_dropout_0.2_new.txt &&
python gnn_louvain.py --network ./dataset/mexican --num_trials 10 --dropout 0.2 > ./output/mexican_dropout_0.2_new.txt &&
python gnn_louvain.py --network ./dataset/dolphin --num_trials 10 --dropout 0.2 > ./output/dolphin_dropout_0.2_new.txt &&
python gnn_louvain.py --network ./dataset/railway --num_trials 10 --dropout 0.2 > ./output/railway_dropout_0.2_new.txt &&
python gnn_louvain.py --network ./dataset/cora --num_trials 10 --dropout 0.2 > ./output/cora_dropout_0.2_new.txt &&
python gnn_louvain.py --network ./dataset/pubmed --num_trials 10 --dropout 0.2 > ./output/pubmed_dropout_0.2_new.txt &

python gnn_louvain.py --network ./syn_dataset/ds/d_avg_10 --num_trials 10 --dropout 0.2 > ./output/syn/d_avg_10.txt &&
python gnn_louvain.py --network ./syn_dataset/ds/d_avg_20 --num_trials 10 --dropout 0.2 > ./output/syn/d_avg_20.txt &&
python gnn_louvain.py --network ./syn_dataset/ds/d_avg_30 --num_trials 10 --dropout 0.2 > ./output/syn/d_avg_30.txt &&
python gnn_louvain.py --network ./syn_dataset/ds/d_max_100 --num_trials 10 --dropout 0.2 > ./output/syn/d_max_100.txt &&
python gnn_louvain.py --network ./syn_dataset/ds/d_max_150 --num_trials 10 --dropout 0.2 > ./output/syn/d_max_150.txt &&
python gnn_louvain.py --network ./syn_dataset/ds/d_max_200 --num_trials 10 --dropout 0.2 > ./output/syn/d_max_200.txt &&
python gnn_louvain.py --network ./syn_dataset/ds/mu_0.1 --num_trials 10 --dropout 0.2 > ./output/syn/mu_0.1.txt &&
python gnn_louvain.py --network ./syn_dataset/ds/mu_0.3 --num_trials 10 --dropout 0.2 > ./output/syn/mu_0.3.txt &&
python gnn_louvain.py --network ./syn_dataset/ds/mu_0.5 --num_trials 10 --dropout 0.2 > ./output/syn/mu_0.5.txt &

python gnn_louvain.py --network ./syn_dataset/scalability/scalability_1000 --num_trials 10 --dropout 0.2 > ./output/syn/scalability_1000.txt &&
python gnn_louvain.py --network ./syn_dataset/scalability/scalability_2000 --num_trials 10 --dropout 0.2 > ./output/syn/scalability_2000.txt &&
python gnn_louvain.py --network ./syn_dataset/scalability/scalability_4000 --num_trials 10 --dropout 0.2 > ./output/syn/scalability_4000.txt &&
python gnn_louvain.py --network ./syn_dataset/scalability/scalability_8000 --num_trials 10 --dropout 0.2 > ./output/syn/scalability_8000.txt &&
python gnn_louvain.py --network ./syn_dataset/scalability/scalability_16000 --num_trials 10 --dropout 0.2 > ./output/syn/scalability_16000.txt &





python gnn_louvain.py --network ./dataset/strike --num_trials 10 --dropout 0.5 > ./output/strike_dropout_0.5_new.txt &&
python gnn_louvain.py --network ./dataset/railway --num_trials 10 --dropout 0.5 > ./output/railway_dropout_0.5_new.txt &&
python gnn_louvain.py --network ./dataset/football --num_trials 10 --dropout 0.5 > ./output/football_dropout_0.5_new.txt &&
python gnn_louvain.py --network ./dataset/dolphin --num_trials 10 --dropout 0.5 > ./output/dolphin_dropout_0.5_new.txt &&
python gnn_louvain.py --network ./dataset/football --num_trials 10 --dropout 0.5 > ./output/football_dropout_0.5_new.txt &&
python gnn_louvain.py --network ./dataset/polblogs --num_trials 10 --dropout 0.5 > ./output/polblogs_dropout_0.5_new.txt &&
python gnn_louvain.py --network ./dataset/cora --num_trials 10 --dropout 0.5 > ./output/cora_dropout_0.5_new.txt &

