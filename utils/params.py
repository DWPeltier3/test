# Copyright 2020, Prof. Marko Orescanin, NPS
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Created by marko.orescanin@nps.edu on 7/21/20

"""params.py

This module contains all parameter handling for the project, including
all command line tunable parameter definitions, any preprocessing
of those parameters, or generation/specification of other parameters.

"""
import argparse
import os
# import yaml


def make_argparser():
    parser = argparse.ArgumentParser(description="Arguments to run training")
    
    parser.add_argument("--mode", type=str, default="train", help="train or predict")

    parser.add_argument("--trained_model", type=str, 
                        default="/home/donald.peltier/swarm/model/swarm_class2023-08-30_15-09-16_fcmc_adam2/model.keras",
                        help="full path to previously trained 'model.keras' to use for predictions")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, help="location of data.npz file")
    # parser.add_argument("--real_data_path", type=str, default=None, help="location of real_data.npz file")
    parser.add_argument("--window", type=int, default=-1, help="observation window; -1 uses full window")

    parser.add_argument("--model_type", type=str, help="fc=fully connect, cn=CNN, fcn=fully CNN, res=ResNet, tr=transformer")
    parser.add_argument("--output_type", type=str, help="mc=multiclass, ml=multilabel, mo=multiout")
    parser.add_argument("--output_length", type=str, help="vec=vector (final only), seq=sequence (every time step)")

    parser.add_argument("--dropout", type=float, default=0, help="dropout percentage 0 to 1 (ie. 0.2=20%)")
    parser.add_argument("--kernel_initializer", type=str, default="glorot_uniform", help="glorot_uniform/normal (default, sigmoid), he_uniform/normal (relu)")
    parser.add_argument("--kernel_regularizer", type=str, default="None", help="l1, l2, l1_l2")

    parser.add_argument("--optimizer", type=str, default="adam", help="SGD optimizer")
    parser.add_argument("--initial_learning_rate", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--callback_list", type=str, default="checkpoint, early_stopping, csv_log, save", help="comma separated callbacks to be added")
    parser.add_argument("--patience", type=int, default=50, help="training epochs with negligible improvement before training stops")
    
    parser.add_argument("--tune_type", type=str, default="r", help="tuner type: r=random, b=bayesian, h=hyperband")
    parser.add_argument("--tune_epochs", type=int, default=100, help="number of epochs used in Keras Tuner")
    
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=50, help="bacth size")
    parser.add_argument("--train_val_split", type=float, default=0.2, help="training validation holdout percentage 0 to 1 (ie. 0.2=20%)")
    # parser.add_argument("--window", nargs='+', type=int, default=-1, help="observation window") #use to implement list of windows to train/evaluate
    # parser.add_argument("--l2",type=float,default=1e-4,help="l2 regularization")
    # parser.add_argument("--test_dir", type=str)
    # parser.add_argument("--train_dir", type=str)
    # parser.add_argument("--val_dir", type=str)
    # parser.add_argument("--checkpoint_path", type=str, help="path to the checkpoint to continue training")
    
    return parser.parse_args()


# you have to use str2bool because of an issue with argparser and bool type
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "1"):
        return True
    elif v.lower() in ("false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_hparams():
    """any preprocessing, special handling of the hparams object"""
    parser = make_argparser()
    print('\n*** PARAMETERS ***')
    print(parser)
    return parser

def make_model_folder(hparams):
    if not os.path.exists(hparams.model_dir):
        os.mkdir(hparams.model_dir)

def save_hparams(hparams):
    make_model_folder(hparams)
    path_ = os.path.join(hparams.model_dir, "params.txt")
    hparams_ = vars(hparams)
    with open(path_, "w") as f:
        for arg in hparams_:
            print(arg, ":", hparams_[arg])
            f.write(arg + ":" + str(hparams_[arg]) + "\n")

    ## Commented out as don't require "params.yml" 
    # path_ = os.path.join(hparams.model_dir, "params.yml")
    # with open(path_, "w") as f:
    #     yaml.dump(
    #         hparams_, f, default_flow_style=False
    #     )  # save hparams as a yaml, since that's easier to read and use
