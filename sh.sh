model_type=wav2vec2-large-robust

model=AuxFormer

corpus=IEMOCAP
num_classes=four #four or ALL
output_num=4
output_dim=4
seed=0
epochs=20
label_type=categorical
label_learning=hard-label
lr=0.00001
hidden_dim=512
num_layers=8
loss_w=0.7
v_dim=50

corpus_type=${corpus}_${num_classes}

fold=1
# Training
python -u train_test.py \
--device            cuda \
--model_type        $model_type \
--lr                $lr \
--corpus_type       $corpus_type \
--seed              $seed \
--epochs            $epochs \
--batch_size        2 \
--hidden_dim        $hidden_dim \
--num_layers        $num_layers \
--output_num        $output_num \
--output_dim        $output_dim \
--fold              $fold \
--loss_w            $loss_w \
--label_type        categorical \
--label_learning    $label_learning \
--corpus            $corpus \
--num_classes       $num_classes \
--model_path        model/${model_type}/${corpus_type}/${label_type}/${label_learning}/${model}/seed_${seed}/fold_${fold}

fold=2
# Training
python -u train_test.py \
--device            cuda \
--model_type        $model_type \
--lr                $lr \
--corpus_type       $corpus_type \
--seed              $seed \
--epochs            $epochs \
--batch_size        2 \
--hidden_dim        $hidden_dim \
--num_layers        $num_layers \
--output_num        $output_num \
--output_dim        $output_dim \
--fold              $fold \
--loss_w            $loss_w \
--label_type        categorical \
--label_learning    $label_learning \
--corpus            $corpus \
--num_classes       $num_classes \
--model_path        model/${model_type}/${corpus_type}/${label_type}/${label_learning}/${model}/seed_${seed}/fold_${fold}

fold=3
# Training
python -u train_test.py \
--device            cuda \
--model_type        $model_type \
--lr                $lr \
--corpus_type       $corpus_type \
--seed              $seed \
--epochs            $epochs \
--batch_size        2 \
--hidden_dim        $hidden_dim \
--num_layers        $num_layers \
--output_num        $output_num \
--output_dim        $output_dim \
--fold              $fold \
--loss_w            $loss_w \
--label_type        categorical \
--label_learning    $label_learning \
--corpus            $corpus \
--num_classes       $num_classes \
--model_path        model/${model_type}/${corpus_type}/${label_type}/${label_learning}/${model}/seed_${seed}/fold_${fold}

fold=4
# Training
python -u train_test.py \
--device            cuda \
--model_type        $model_type \
--lr                $lr \
--corpus_type       $corpus_type \
--seed              $seed \
--epochs            $epochs \
--batch_size        2 \
--hidden_dim        $hidden_dim \
--num_layers        $num_layers \
--output_num        $output_num \
--output_dim        $output_dim \
--fold              $fold \
--loss_w            $loss_w \
--label_type        categorical \
--label_learning    $label_learning \
--corpus            $corpus \
--num_classes       $num_classes \
--model_path        model/${model_type}/${corpus_type}/${label_type}/${label_learning}/${model}/seed_${seed}/fold_${fold}

fold=5
# Training
python -u train_test.py \
--device            cuda \
--model_type        $model_type \
--lr                $lr \
--corpus_type       $corpus_type \
--seed              $seed \
--epochs            $epochs \
--batch_size        2 \
--hidden_dim        $hidden_dim \
--num_layers        $num_layers \
--output_num        $output_num \
--output_dim        $output_dim \
--fold              $fold \
--loss_w            $loss_w \
--label_type        categorical \
--label_learning    $label_learning \
--corpus            $corpus \
--num_classes       $num_classes \
--model_path        model/${model_type}/${corpus_type}/${label_type}/${label_learning}/${model}/seed_${seed}/fold_${fold}
