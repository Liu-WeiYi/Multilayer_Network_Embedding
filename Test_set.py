#!/usr/bin/python
# -*- coding: utf-8 -*-

import Reader, Sampler, Evaluator, Node2Vec, Word2Vec, Node2Vec_LayerSelect, Evaluator_has, link_pred, sys, NMI_Calculation, numpy, pickle, itertools
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import f1_score

def common_nodes(mapping, Original_graphs):
    n_list = []
    for node in mapping.keys():
        for g in Original_graphs:
            if node in g.nodes():
                n_list.append(node)
                break
    return set(n_list)

def clustering(c_t):
    ct_dict = {}
    for e in range(len(c_t)):
        if c_t[e][0] not in ct_dict.keys():
            ct_dict[c_t[e][0]] = [e]
        else:
            ct_dict[c_t[e][0]].append(e)
    c_true = []
    for x in ct_dict.keys():
        c_true.append(ct_dict[x])
    return c_true



class C_N_J:
    def __init__(self, path, sampling_p):
        self.path = path
        self.s_p = sampling_p

    def run(self):
        path = self.path
        ori_g = Reader.single_readG(path)
        _, samp_g = Sampler.single_sampling(path, self.s_p)

        p = link_pred.Prediction()
        v_set = p.create_vertex(ori_g.edges())
        matrix_ori = p.create_adjmatrix(ori_g.edges(), v_set)
        matrix_samp = p.create_adjmatrix(samp_g.edges(), v_set)
        cn = link_pred.CommonNeighbors()
        score_cn = cn.fit(matrix_ori)
        auc_cn = p.auc_score(score_cn, matrix_ori, matrix_samp, 'cc')
        print("*** CommonNeighbors AUC:", auc_cn)

        ja = link_pred.Jaccard()
        score_ja = ja.fit(matrix_ori)
        auc_ja = p.auc_score(score_ja, matrix_ori, matrix_samp, 'cc')
        print("*** Jaccard AUC:", auc_ja)

class Mergeing_vec_N2V:
    def __init__(self, path, sampling_p, examing_p, p, q, num_walks, walk_length, flag, r):
        self.path = path
        self.s_p = sampling_p
        self.e_p = examing_p
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.flag = flag
        self.r = r

    def run(self):
        path = self.path
        #### Step 1: reading and sampling graphs

        m_graph, nx_graphs, total_edges = Reader.multi_readG_with_Merg(path)
        print("%d total nodes"%len(m_graph.nodes()))
        r_list, m_graph_sampled, nx_graphs_sampled = Sampler.multi_sampling_with_Merg(path, self.s_p)
        print("%d edges before sampling, %d edges after sampling. sampled %d "%(len(m_graph.edges()), len(m_graph_sampled.edges()), len(r_list)))

        r_set = set([node for edge in r_list for node in edge])

        if self.flag == 0 or self.flag == 1:

        #### Step 2: Aggregated graph
            #for i in range(2):

            M_G = Node2Vec.Graph(m_graph_sampled, self.p, self.q)
            M_G.preprocess_transition_probs()
            M_walks = M_G.simulate_walks(self.num_walks, self.walk_length)

            M_words = []
            for walk in M_walks:
                M_words.extend([str(step) for step in walk])

            M_L = Word2Vec.Learn(M_words)
            M_matrix, M_mapping = M_L.train()

            eval_p = Evaluator.Precision_Eval(M_matrix, M_mapping, m_graph, r_set, self.e_p)
            precision, recall, F = eval_p.eval()
            print("*** Aggregated graph: precision %f, accuracy %f, F %f "%(precision, recall, F))


            eval_a = Evaluator.AUC_Eval(M_matrix, M_mapping, m_graph, m_graph_sampled)
            M_auc = eval_a.eval_auc(1)
            print("@@@ Merged graph AUC:", M_auc)

            print("-----------------------DONE--------------------------------")
        #### Step 3: Aggregated result

        if self.flag == 0 or self.flag == 2:

            T_matrix = {}
            T_mapping = {}
            for g in nx_graphs_sampled:
                #print(g.edges())
                G = Node2Vec.Graph(g, self.p, self.q)
                G.preprocess_transition_probs()
                walks = G.simulate_walks(self.num_walks, self.walk_length)
                words = []
                for walk in walks:
                    words.extend([str(step) for step in walk])

                L = Word2Vec.Learn(words)
                matrix, mapping = L.train()
                T_matrix[g] = matrix
                T_mapping[g] = mapping

            eval_p_s = Evaluator.combining_Precision_Eval(T_matrix, T_mapping, nx_graphs, r_set, self.e_p)
            precision, recall, F = eval_p_s.eval()
            print("*** Aggregated result: precision %f, accuracy %f, F %f"%(precision, recall, F))

            eval_a = Evaluator.combining_AUC_Eval(T_matrix, T_mapping, nx_graphs, nx_graphs_sampled)
            S_auc = eval_a.eval_auc(1)
            print('@@@ Separated garph AUC:', S_auc)

            print("-----------------------DONE--------------------------------")

        #### Step 4: MKII verification

        if self.flag == 0 or self.flag == 3:
            graph_list_sampled=[]
            graph_list_sampled.append(m_graph_sampled)
            graph_list = []
            graph_list.append(m_graph)
            w_dict = {}
            MK_G = Node2Vec_LayerSelect.Graph(graph_list, self.p, self.q, self.r)
            MK_G.preprocess_transition_probs(w_dict, 1)
            MK_walks = MK_G.simulate_walks(self.num_walks, self.walk_length)

            MK_words = []
            for walk in MK_walks:
                MK_words.extend([str(step) for step in walk])

            M_L = Word2Vec.Learn(MK_words)
            M_matrix, M_mapping = M_L.train()

            eval_p = Evaluator.Precision_Eval(M_matrix, M_mapping, graph_list[0], r_set, self.e_p)
            precision, recall, F = eval_p.eval()
            print("*** MKII verification: precision %f, accuracy %f, F %f"%(precision, recall, F))


            eval_a = Evaluator.AUC_Eval(M_matrix, M_mapping, m_graph, m_graph_sampled)
            M_auc = eval_a.eval_auc(1)
            print("@@@ Merged graph AUC:", M_auc)

            print("-----------------------DONE--------------------------------")

        #### Step 5: MKII Random
        if self.flag == 0 or self.flag == 4:
            w_dict = Reader.weight(self.path)
            #print(w_dict)

            MK_G = Node2Vec_LayerSelect.Graph(nx_graphs_sampled, self.p, self.q, self.r)
            MK_G.preprocess_transition_probs(w_dict, 1)
            MK_walks = MK_G.simulate_walks(self.num_walks, self.walk_length)

            MK_words = []
            for walk in MK_walks:
                MK_words.extend([str(step) for step in walk])

            M_L = Word2Vec.Learn(MK_words)
            M_matrix, M_mapping = M_L.train()

            eval_p = Evaluator.Precision_Eval(M_matrix, M_mapping, nx_graphs, r_set, self.e_p)
            precision, recall, F = eval_p.eval()
            print("*** MKII Random: precision %f, accuracy %f, F %f"%(precision, recall, F))


            eval_a = Evaluator.AUC_Eval(M_matrix, M_mapping, nx_graphs, nx_graphs_sampled)
            M_auc = eval_a.eval_auc(1)
            print("@@@ MKII Random AUC:", M_auc)

            print("-----------------------DONE--------------------------------")

        #### Step 6: MKII Weighted
        if self.flag == 0 or self.flag == 4:
            w_dict = Reader.weight(self.path)
            #print(w_dict)

            MK_G = Node2Vec_LayerSelect.Graph(nx_graphs_sampled, self.p, self.q, self.r)
            MK_G.preprocess_transition_probs(w_dict, 2)
            MK_walks = MK_G.simulate_walks(self.num_walks, self.walk_length)

            MK_words = []
            for walk in MK_walks:
                MK_words.extend([str(step) for step in walk])

            M_L = Word2Vec.Learn(MK_words)
            M_matrix, M_mapping = M_L.train()

            eval_p = Evaluator.Precision_Eval(M_matrix, M_mapping, nx_graphs, r_set, self.e_p)
            precision, recall, F = eval_p.eval()
            print("*** MKII Weighted: precision %f, accuracy %f, F %f"%(precision, recall, F))

            eval_a = Evaluator.AUC_Eval(M_matrix, M_mapping, nx_graphs, nx_graphs_sampled)
            M_auc = eval_a.eval_auc(1)
            print("@@@ MKII Weighted AUC:", M_auc)


            print("-----------------------DONE--------------------------------")

        #### Step 7: MKII Biased
        if self.flag == 0 or self.flag == 4:
            w_dict = Reader.weight(self.path)
            #print(w_dict)

            MK_G = Node2Vec_LayerSelect.Graph(nx_graphs_sampled, self.p, self.q, self.r)
            MK_G.preprocess_transition_probs(w_dict, 0)
            MK_walks = MK_G.simulate_walks(self.num_walks, self.walk_length)

            MK_words = []
            for walk in MK_walks:
                MK_words.extend([str(step) for step in walk])

            M_L = Word2Vec.Learn(MK_words)
            M_matrix, M_mapping = M_L.train()

            eval_p = Evaluator.Precision_Eval(M_matrix, M_mapping, nx_graphs, r_set, self.e_p)
            precision, recall, F = eval_p.eval()
            print("*** MKII Biased: precision %f, accuracy %f, F %f"%(precision, recall, F))
            eval_a = Evaluator.AUC_Eval(M_matrix, M_mapping, nx_graphs, nx_graphs_sampled)
            M_auc = eval_a.eval_auc(1)
            print("@@@ MKII Biased AUC:", M_auc)

            print("-----------------------DONE--------------------------------")

        #### Step 8: MKII Biased_ii
        if self.flag == 0 or self.flag == 4:
            w_dict = Reader.weight(self.path)
            #print(w_dict)

            MK_G = Node2Vec_LayerSelect.Graph(nx_graphs_sampled, self.p, self.q, self.r)
            MK_G.preprocess_transition_probs(w_dict, 3)
            MK_walks = MK_G.simulate_walks(self.num_walks, self.walk_length)

            MK_words = []
            for walk in MK_walks:
                MK_words.extend([str(step) for step in walk])

            M_L = Word2Vec.Learn(MK_words)
            M_matrix, M_mapping = M_L.train()

            eval_p = Evaluator.Precision_Eval(M_matrix, M_mapping, nx_graphs, r_set, self.e_p)
            precision, recall, F = eval_p.eval()
            print("*** MKII Biased_ii: precision %f, accuracy %f, F %f"%(precision, recall, F))
            eval_a = Evaluator.AUC_Eval(M_matrix, M_mapping, nx_graphs, nx_graphs_sampled)
            M_auc = eval_a.eval_auc(1)
            print("@@@ MKII Biased_ii AUC:", M_auc)

            print("-----------------------DONE--------------------------------")

        if self.flag == 4:

            for r in range(11):

                r_t = r/10.0

                if r_t == 0:
                    w_dict = Reader.weight(self.path)
                    #print(w_dict)

                    MK_G = Node2Vec_LayerSelect.Graph(nx_graphs_sampled, self.p, self.q, 0.1)
                    MK_G.preprocess_transition_probs(w_dict, 1)
                    MK_walks = MK_G.simulate_walks(self.num_walks, self.walk_length)

                    MK_words = []
                    for walk in MK_walks:
                        MK_words.extend([str(step) for step in walk])

                    M_L = Word2Vec.Learn(MK_words)
                    M_matrix, M_mapping = M_L.train()

                    eval_p = Evaluator.Precision_Eval(M_matrix, M_mapping, nx_graphs, r_set, self.e_p)
                    precision, recall, F = eval_p.eval()
                    print("*** MKII Random: precision %f, accuracy %f, F %f"%(precision, recall, F))
                    eval_a = Evaluator.AUC_Eval(M_matrix, M_mapping, nx_graphs, nx_graphs_sampled)
                    M_auc = eval_a.eval_auc(1)
                    print("@@@ MKII Random AUC:", M_auc)

                    print("-----------------------DONE--------------------------------")

                else:
                    w_dict = Reader.weight(self.path)
                    #print(w_dict)

                    MK_G = Node2Vec_LayerSelect.Graph(nx_graphs_sampled, self.p, self.q, r_t)
                    MK_G.preprocess_transition_probs(w_dict, 3)
                    MK_walks = MK_G.simulate_walks(self.num_walks, self.walk_length)

                    MK_words = []
                    for walk in MK_walks:
                        MK_words.extend([str(step) for step in walk])

                    M_L = Word2Vec.Learn(MK_words)
                    M_matrix, M_mapping = M_L.train()

                    eval_p = Evaluator.Precision_Eval(M_matrix, M_mapping, nx_graphs, r_set, self.e_p)
                    precision, recall, F = eval_p.eval()
                    print("*** MKII Biased_ii with %f: precision %f, accuracy %f, F %f"%(r_t, precision, recall, F))
                    eval_a = Evaluator.AUC_Eval(M_matrix, M_mapping, nx_graphs, nx_graphs_sampled)
                    M_auc = eval_a.eval_auc(1)
                    print("@@@ MKII Biased_ii AUC:", M_auc)


        #### Step 9: CommoneNeighbors and Jaccard
        if self.flag == 0 or self.flag == 5:
            p = link_pred.Prediction()
            v_set = p.create_vertex(m_graph.edges())
            matrix_perm = p.create_adjmatrix([edge for edge in itertools.combinations(r_set, 2)], v_set)
            matrix_ori = p.create_adjmatrix(m_graph.edges(), v_set)
            matrix_samp = p.create_adjmatrix(m_graph_sampled.edges(), v_set)
            cn = link_pred.CommonNeighbors()
            score_cn = cn.fit(matrix_ori)
            C_precision, C_recall, C_F= p.acc(score_cn, matrix_ori, matrix_perm, self.e_p)
            print("*** CommonNeighbors: precision %f, accuracy %f, F %f"%(C_precision, C_recall, C_F))
            C_auc = p.auc_score(score_cn, matrix_ori, matrix_samp, "cc")
            print("@@@ CommonNeighbors: AUC %f", C_auc)

            ja = link_pred.Jaccard()
            score_ja = ja.fit(matrix_ori)
            J_precision, J_recall, J_F= p.acc(score_ja, matrix_ori, matrix_perm, self.e_p)
            print("*** Jaccard: precision %f, accuracy %f, F %f"%(J_precision, J_recall, J_F))
            J_auc = p.auc_score(score_ja, matrix_ori, matrix_samp, "cc")
            print("@@@ Jaccard: AUC %f", J_auc)
            print("-----------------------DONE--------------------------------")


class N2V_layer_selection:
    def __init__(self, path, sampling_p, examing_p, p, q, num_walks, walk_length):
        self.path = path
        self.s_p = sampling_p
        self.e_p = examing_p
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length

    def run(self):
        path = self.path
        nx_graphs, total_edges = Reader.multi_readG(path)
        r_list, nx_graphs_sampled = Sampler.multi_sampling(path, self.s_p)
        print('%d edges sampled, graph length is %d'%(len(r_list), len(nx_graphs_sampled)))
        MK_G = Node2Vec_LayerSelect.Graph(nx_graphs_sampled, self.p, self.q)
        MK_G.preprocess_transition_probs()
        MK_walks = MK_G.simulate_walks(self.num_walks, self.walk_length)

        MK_words = []
        for walk in MK_walks:
            MK_words.extend([str(step) for step in walk])


        M_L = Word2Vec.Learn(MK_words)
        M_matrix, M_mapping = M_L.train()

        r_set = set([node for edge in r_list for node in edge])

        eval_p = Evaluator.Precision_Eval(M_matrix, M_mapping, nx_graphs, r_set, self.e_p)
        M_precision = eval_p.eval()
        print("*** Merged graph precision: ", M_precision)

class N2V_on_off:
    def __init__(self, path, sampling_p, examing_p, p, q, num_walks, walk_length, flag):
        self.path = path
        self.s_p = sampling_p
        self.e_p = examing_p
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.flag = flag

    def run(self):
        path = self.path
        online_dir = path+"online/"
        online_graphs, _ = Reader.multi_readG(online_dir)
        offline_dir = path+"offline/"
        offline_graphs, _ = Reader.multi_readG(offline_dir)

        ### Step 1: learing with N2V MKII
        if self.flag == 0 or self.flag == 1:
            off_G = Node2Vec_LayerSelect.Graph(offline_graphs, self.p, self.q)
            off_G.preprocess_transition_probs()
            off_walks = off_G.simulate_walks(self.num_walks, self.walk_length)

            off_words = []
            for walk in off_walks:
                off_words.extend([str(step) for step in walk])


            off_L = Word2Vec.Learn(off_words)
            off_matrix, off_mapping = off_L.train()

            on_G = Node2Vec_LayerSelect.Graph(online_graphs, self.p, self.q)
            on_G.preprocess_transition_probs()
            on_walks = on_G.simulate_walks(self.num_walks, self.walk_length)

            on_words = []
            for walk in on_walks:
                on_words.extend([str(step) for step in walk])


            on_L = Word2Vec.Learn(on_words)
            on_matrix, on_mapping = on_L.train()

            off_perm_list = common_nodes(off_mapping, online_graphs)


            off_eval = Evaluator.Precision_Eval(off_matrix, off_mapping, online_graphs, off_perm_list, self.e_p)
            off_precision = off_eval.eval()
            print("*** Off to on MKII precision: ", off_precision)

            off_eval_a = Evaluator.AUC_Eval(off_matrix, off_mapping, online_graphs, offline_graphs)
            off_auc = off_eval_a.eval_auc(0)
            print("@@@ Off to on MKII AUC:", off_auc)

            on_perm_list = common_nodes(on_mapping, offline_graphs)

            on_eval = Evaluator.Precision_Eval(on_matrix, on_mapping, offline_graphs, on_perm_list, self.e_p)
            on_precision = on_eval.eval()
            print("*** On to off MKII precision: ", on_precision)

            on_eval_a = Evaluator.AUC_Eval(on_matrix, on_mapping, offline_graphs, online_graphs)
            on_auc = on_eval_a.eval_auc(0)
            print("@@@ On to off MKII AUC:", on_auc)

        if self.flag == 0 or self.flag == 2:
            on_matrix = {}
            on_mapping = {}
            on_perm_list = []
            for g in online_graphs:
                G = Node2Vec.Graph(g, self.p, self.q)
                G.preprocess_transition_probs()
                walks = G.simulate_walks(self.num_walks, self.walk_length)
                words = []
                for walk in walks:
                    words.extend([str(step) for step in walk])

                L = Word2Vec.Learn(words)
                matrix, mapping = L.train()
                on_matrix[g] = matrix
                on_mapping[g] = mapping
                on_perm_list.extend(common_nodes(mapping, offline_graphs))

            on_perm_list = set([node for node in on_perm_list])
            #print(on_perm_list)
            #print(on_mapping)
            eval_p_on = Evaluator.combining_Precision_Eval(on_matrix, on_mapping, offline_graphs, on_perm_list, self.e_p)
            on_precision = eval_p_on.eval()
            print("*** on to off precision: ", on_precision)

            on_eval_a = Evaluator.combining_AUC_Eval(on_matrix, on_mapping, offline_graphs, online_graphs)
            on_auc = on_eval_a.eval_auc(0)
            print("@@@ On to off  AUC:", on_auc)

            off_matrix = {}
            off_mapping = {}
            off_perm_list = []
            for g in offline_graphs:
                G = Node2Vec.Graph(g, self.p, self.q)
                G.preprocess_transition_probs()
                walks = G.simulate_walks(self.num_walks, self.walk_length)
                words = []
                for walk in walks:
                    words.extend([str(step) for step in walk])

                L = Word2Vec.Learn(words)
                matrix, mapping = L.train()
                off_matrix[g] = matrix
                off_mapping[g] = mapping
                off_perm_list.extend(common_nodes(mapping, online_graphs))

            off_perm_list = set([node for node in off_perm_list])
            eval_p_off = Evaluator.combining_Precision_Eval(off_matrix, off_mapping, online_graphs, off_perm_list, self.e_p)
            off_precision = eval_p_off.eval()
            print("*** off to on precision: ", off_precision)

            off_eval_a = Evaluator.combining_AUC_Eval(off_matrix, off_mapping, online_graphs, offline_graphs)
            off_auc = off_eval_a.eval_auc(0)
            print("@@@ Off to on  AUC:", off_auc)




class Community:
    def __init__(self, path, p, q, num_walks, walk_length):
        self.path = path
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length

    def run(self, flag):
        nx_graphs, _ = Reader.multi_readG(self.path)

        if flag == "LN":
            r_t = Reader.true_cluster(self.path).tolist()
            print(clustering(r_t))
            cluster_true = [r[0]-1 for r in r_t]
            k_list = [k for k in range(2,11)]
        else:
            cluster_true = []
            k_list = [2,3,6,8]
            for i in range(29):
                if i < 12:
                    cluster_true.append(0)
                else:
                    cluster_true.append(1)

        w_dict = Reader.weight(self.path)
        print(nx_graphs[0])
        MK_G = Node2Vec_LayerSelect.Graph(nx_graphs, self.p, self.q, 0.5)
        MK_G.preprocess_transition_probs(w_dict, 2)
        MK_walks = MK_G.simulate_walks(self.num_walks, self.walk_length)

        MK_words = []
        for walk in MK_walks:
            MK_words.extend([str(step) for step in walk])


        M_L = Word2Vec.Learn(MK_words)
        M_matrix, M_mapping = M_L.train()

        result = {}
        for k in k_list:
            cluster_trained = KMeans(n_clusters=k, random_state=0).fit_predict(M_matrix).tolist()

            length = min(len(cluster_true),len(cluster_trained))

            r = normalized_mutual_info_score(cluster_true[0:length], cluster_trained[0:length])
            f = f1_score(cluster_true[0:length], cluster_trained[0:length], average='micro')
            print(cluster_trained)
            print(cluster_true)

            result[k] = (r, f)
            #pickle.dump(cluster_trained, open(self.path+str(k)+'.pickle', '+wb'))


        print(result)


class change_r:
    def __init__(self, path, sampling_p, examing_p, p, q, num_walks, walk_length):
        self.path = path
        self.s_p = sampling_p
        self.e_p = examing_p
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length

    def run(self):
        path = self.path
        #### Step 1: reading and sampling graphs
        '''
        m_graph, nx_graphs, total_edges = Reader.multi_readG_with_Merg(path)
        print("%d total nodes"%len(m_graph.nodes()))
        r_list, m_graph_sampled, nx_graphs_sampled = Sampler.multi_sampling_with_Merg(path, self.s_p)
        print("%d edges before sampling, %d edges after sampling. sampled %d "%(len(m_graph.edges()), len(m_graph_sampled.edges()), len(r_list)))

        r_set = set([node for edge in r_list for node in edge])
        '''
        nx_graphs_sampled, _ = Reader.multi_readG(self.path)
        cluster_true = []
        for i in range(29):
            if i < 12:
                cluster_true.append(0)
            else:
                cluster_true.append(1)


        for r in range(11):

            r_t = r/10.0

            if r_t == 0:
                w_dict = Reader.weight(self.path)
                #print(w_dict)

                MK_G = Node2Vec_LayerSelect.Graph(nx_graphs_sampled, self.p, self.q, 0.1)
                MK_G.preprocess_transition_probs(w_dict, 1)
                MK_walks = MK_G.simulate_walks(self.num_walks, self.walk_length)

                MK_words = []
                for walk in MK_walks:
                    MK_words.extend([str(step) for step in walk])

                M_L = Word2Vec.Learn(MK_words)
                M_matrix, M_mapping = M_L.train()
                '''
                eval_p = Evaluator.Precision_Eval(M_matrix, M_mapping, nx_graphs, r_set, self.e_p)
                precision, recall, F = eval_p.eval()
                print("*** MKII Biased: precision %f, accuracy %f, F %f"%(precision, recall, F))
                eval_a = Evaluator.AUC_Eval(M_matrix, M_mapping, nx_graphs, nx_graphs_sampled)
                M_auc = eval_a.eval_auc(1)
                print("@@@ MKII Biased AUC:", M_auc)
                '''


            else:
                w_dict = Reader.weight(self.path)
                #print(w_dict)

                MK_G = Node2Vec_LayerSelect.Graph(nx_graphs_sampled, self.p, self.q, r_t)
                MK_G.preprocess_transition_probs(w_dict, 3)
                MK_walks = MK_G.simulate_walks(self.num_walks, self.walk_length)

                MK_words = []
                for walk in MK_walks:
                    MK_words.extend([str(step) for step in walk])

                M_L = Word2Vec.Learn(MK_words)
                M_matrix, M_mapping = M_L.train()
                '''
                eval_p = Evaluator.Precision_Eval(M_matrix, M_mapping, nx_graphs, r_set, self.e_p)
                precision, recall, F = eval_p.eval()
                print("*** MKII Biased_ii with %f: precision %f, accuracy %f, F %f"%(r_t, precision, recall, F))
                eval_a = Evaluator.AUC_Eval(M_matrix, M_mapping, nx_graphs, nx_graphs_sampled)
                M_auc = eval_a.eval_auc(1)
                print("@@@ MKII Biased_ii AUC:", M_auc)
                '''

            cluster_trained = KMeans(n_clusters=2, random_state=0).fit_predict(M_matrix).tolist()

            length = min(len(cluster_true),len(cluster_trained))

            r = normalized_mutual_info_score(cluster_true[0:length], cluster_trained[0:length])
            mi_f = f1_score(cluster_true[0:length], cluster_trained[0:length], average='micro')
            ma_f = f1_score(cluster_true[0:length], cluster_trained[0:length], average='macro')
            print("r is %f: nmi %f, micro_f %f, macro_f %f"%(r_t, r, mi_f, ma_f))
            print("-----------------------DONE--------------------------------")

class Airline:
    def __init__(self, path, sampling_p, examing_p, p, q, num_walks, walk_length, flag, r):
        self.path = path
        self.e_p = examing_p
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.flag = flag
        self.r = r

    def run(self):
        path = self.path
        #### Step 1: reading and sampling graphs

        nx_graphs, airport_mapping, airport_dst = Reader.read_airline(path)
        print(nx_graphs[0].nodes())

        r_set=set()

        if self.flag == 0 or self.flag == 4:
            w_dict = {}

            MK_G = Node2Vec_LayerSelect.Graph(nx_graphs, self.p, self.q, self.r)
            MK_G.preprocess_transition_probs(w_dict, 1)
            MK_walks = MK_G.simulate_walks(self.num_walks, self.walk_length)

            MK_words = []
            for walk in MK_walks:
                MK_words.extend([str(step) for step in walk])

            M_L = Word2Vec.Learn(MK_words)
            M_matrix, M_mapping = M_L.train()

            eval_p = Evaluator.Precision_Eval(M_matrix, M_mapping, nx_graphs, r_set, self.e_p)
            precision, recall, F = eval_p.edge_list_eval(airport_dst, airport_mapping)
            print("*** MKII Random: precision %f, accuracy %f, F %f"%(precision, recall, F))

            '''
            eval_a = Evaluator.AUC_Eval(M_matrix, M_mapping, nx_graphs, nx_graphs_sampled)
            M_auc = eval_a.eval_auc(1)
            print("@@@ MKII Random AUC:", M_auc)
            '''
            print("-----------------------DONE--------------------------------")
