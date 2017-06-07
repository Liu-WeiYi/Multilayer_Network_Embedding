import Test_set
import argparse


################# flag reading #######################################
parser = argparse.ArgumentParser()
parser.add_argument('--graph', help='input one file for runing the test')
parser.add_argument('--directory', help='a folder that contains all the graphs for multilayer test')
parser.add_argument('--walk_p', type=float, default=0.5, help='p probability for Node2Vec biased walking')
parser.add_argument('--walk_q', type=float, default=0.5, help='q probability for Node2Vec biased walking')
parser.add_argument('--num_walks', type=int, default=10, help='number of walks to for Node2Vec learning')
parser.add_argument('--walk_length', type=int, default=80, help='the walk length for each walk for Node2Vec learning')
parser.add_argument('--sample_portion', type=float, default=0.1, help='portion of the graph that need to be sampled')
parser.add_argument('--examing_portion', type=float, default=0.1, help='portion of the outcome that need to be compared to get precision')
parser.add_argument('--flag', type=int, default=0, help='control number stages of the test')
parser.add_argument('--MN', action='store_true', help='Node2Vec Multilayer sampling test')
parser.add_argument('--MNMK', action='store_true')
parser.add_argument('--MNOF', action='store_true', help='Node2Vec Multilayer Online Offline test')
parser.add_argument('--LN', action='store_true', help='run community detection on LN network')
parser.add_argument('--VC', action='store_true', help='run community detection on VC network')
parser.add_argument('--R', action='store_true', help='run r from 0.0 - 1.0')
parser.add_argument('--AIR', action='store_true', help='run airline test')
parser.add_argument('--COM', action='store_true', help='CommonNeighbors and Jaccard AUC test')
args = parser.parse_args()

if args.MN:
    print("Running MN")
    MN = Test_set.Mergeing_vec_N2V(args.directory, args.sample_portion, args.examing_portion, args.walk_p, args.walk_q, args.num_walks, args.walk_length, args.flag, 0.5)
    MN.run()

if args.MNOF:
    print("Running MNOF")
    MNOF = Test_set.N2V_on_off(args.directory, args.sample_portion, args.examing_portion, args.walk_p, args.walk_q, args.num_walks, args.walk_length, args.flag)
    MNOF.run()

if args.MNMK:
    print("Running MNMK")
    MNMK = Test_set.N2V_layer_selection(args.directory, args.sample_portion, args.examing_portion, args.walk_p, args.walk_q, args.num_walks, args.walk_length)
    MNMK.run()

if args.COM:
    print("Running CommonNeighbors and Jaccard to verify ACU Algorithm")
    CNJ = Test_set.C_N_J(args.graph, args.sample_portion)
    CNJ.run()

if args.LN:
    print("Running community density LN")
    C = Test_set.Community(args.directory, args.walk_p, args.walk_q, args.num_walks, args.walk_length)
    C.run("LN")

if args.VC:
    print("Running community density VC")
    C = Test_set.Community(args.directory, args.walk_p, args.walk_q, args.num_walks, args.walk_length)
    C.run("VC")

if args.R:
    print("Running for R")
    change_R = Test_set.change_r(args.directory, args.sample_portion, args.examing_portion, args.walk_p, args.walk_q, args.num_walks, args.walk_length)
    change_R.run()

if args.AIR:
    print("Running for Airline test")
    AIR_L = Test_set.Airline(args.directory, args.sample_portion, args.examing_portion, args.walk_p, args.walk_q, args.num_walks, args.walk_length, args.flag, 0.5)
    AIR_L.run()
