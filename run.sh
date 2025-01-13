rs=0 #random seed
dataset=cifar10 # dataset (Check data_loader folder)
model=resnet56 # models (Check model.py)
learning_rate=0.001
weight_decay=1e-5
batch_size=64
client_optimizer="adam"
comm_round=100
num_parties=4 # number of parties
test_comm_round=$((comm_round - 1))
save_models=True
save_client=0 # which client you want to audit. If you want to audit all clients, you can set it to all.
target_round=99 #on which round, you want to audit the privacy risk
target_model_type=global # indicate the model type you want to audit: global or local
start_round=0 # start from which round, you want to audit the privacy risk

python 1_create_split.py data=$dataset random_seed=$rs fl.num_parties=$num_parties
python 2_run_fl.py fl.num_parties=$num_parties data=$dataset random_seed=$rs fl.model=$model fl.learning_rate=$learning_rate fl.weight_decay=$weight_decay fl.train_batch_size=$batch_size fl.batch_size=$batch_size fl.client_optimizer=$client_optimizer fl.comm_round=$comm_round save_models=$save_models save_client=$save_client
python 3_run_audit.py fl.num_parties=$num_parties data=$dataset audit.party=$save_client random_seed=$rs fl.model=$model fl.learning_rate=$learning_rate fl.weight_decay=$weight_decay fl.train_batch_size=$batch_size fl.batch_size=$batch_size fl.client_optimizer=$client_optimizer fl.comm_round=$comm_round save_models=$save_models save_client=$save_client audit.target_round=$target_round audit.start_round=$start_round audit.target_model_type=$target_model_type
