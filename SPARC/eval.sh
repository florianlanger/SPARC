
# Set this path to path of the SPARC data
base_path=/scratch2/fml35/datasets/own_data/sparc_release

echo "Sparc net data download base path" $base_path

network_path=$base_path/main_experiment/sparc_net.pth
out_dir=$base_path/main_experiment/outputs

refine=1
eval_method=init_for_classification
for rot_index in 0 1 2 3
    do
        python eval.py $network_path $refine $rot_index $eval_method $out_dir
    done

python evaluate/combine_classification_preds.py $out_dir

refine=3
python eval.py $network_path $refine 0 init_from_best_rotation_index $out_dir


