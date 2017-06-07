#include<iostream>
#include<map>
#include<vector>
#include<cmath>
#include <fstream>
#include "networks.h"
using namespace std;

#define pi 3.1415926535897932384626433832795


// 控制参数
int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    return -1;
}

//normal distribution
double rand(double min, double max)
{
    return min+(max-min)*rand()/(RAND_MAX+1.0);
}
double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}
double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do{
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }while(dScope>y);
    return x;
//    srand(time(NULL));
//    double pRandomValue = (double)(rand()/(double)RAND_MAX);
//    pRandomValue = pRandomValue*(max-min)+min;
//    return pRandomValue;
}

double sqr(double x)
{
    return x*x;
}

// 获取向量长度
double vec_len(vector<double> &a)
{
    double res=0;
    for (int i=0; i<a.size(); i++)
        res+=a[i]*a[i];
    res = sqrt(res);
    return res;
}

class Train{

public:
    map<pair<int,int>, map<int,int> > ok;
    void add(int x,int y,int z)
    {
        src_list.push_back(x);
        edge_list.push_back(z);
        dst_list.push_back(y);
        ok[make_pair(x,z)][y]=1;
    }

    void prepare(string nodeF, string edgeF, string factF, bool L1, int generate_flag)
    {
        this->L1_flag = L1;
        layer.setName(network_name);
        this->generate_flag = generate_flag;

        // 1. read node file
        ifstream inNodeFile;
        inNodeFile.open(nodeF);
        // 首先判断 文件是否能够打开。。。不能直接退出！
        if (!inNodeFile.is_open()){
            cout << "Could NOT open the file: " << nodeF << endl;
            cout << "Program terminating...\n";
            exit(EXIT_FAILURE);
        }
        while (inNodeFile.good()){
            string src;
            int id;
            inNodeFile >> src;
            inNodeFile >> id;
            if (src != ""){
                node2id[src] = id;
                id2node[id] = src;
                node_num++;
            }

        }
        inNodeFile.close();

        // 2. read edge file
        ifstream inEdgeFile;
        inEdgeFile.open(edgeF);
        // 首先判断 文件是否能够打开。。。不能直接退出！
        if (!inEdgeFile.is_open()){
            cout << "Could NOT open the file: " << edgeF << endl;
            cout << "Program terminating...\n";
            exit(EXIT_FAILURE);
        }
        while (inEdgeFile.good()){
            string edge;
            int id;
            inEdgeFile >> edge;
            inEdgeFile >> id;
            if (edge != ""){
                edge2id[edge] = id;
                id2edge[id] = edge;
                edge_num++;
            }
        }
        inEdgeFile.close();

        // 3. read fact file
        ifstream inFactFile;
        inFactFile.open(factF);
        // 首先判断 文件是否能够打开。。。不能直接退出！
        if (!inFactFile.is_open()){
            cout << "Could NOT open the file: " << factF << endl;
            cout << "Program terminating...\n";
            exit(EXIT_FAILURE);
        }

        vector<std::string> node_list;
        while (inFactFile.good()){
            string src,dst,edge;
            inFactFile >> src;
            inFactFile >> dst;
            inFactFile >> edge;

            // 添加网络节点
            node_list.push_back(src);
            node_list.push_back(dst);
            // 添加该条边到网络中
            layer.add_edge(src,dst,1.0, false);

            if (src != ""){
                if (edge2id.count(edge) == 0){
                    edge2id[edge] = edge_num;
                    edge_num++;
                }
                this->add(node2id[src],node2id[dst],edge2id[edge]);
            }
        }
        layer.add_nodes_from(node_list);
        inFactFile.close();

        cout<<"edge_num="<<edge_num<<endl;
        cout<<"node_num="<<node_num<<endl;
    }

    void run(int dim,double rate_in,double margin_in,string network_name,int sample_number)
    {
        // n_in: dimension
        // rate_in: learning rate
        // margin_in: margin
        // metho_in: network_name
        this->n = dim;
        this->rate = rate_in;
        this->margin = margin_in;
        this->network_name = network_name;
        this->sample_number = sample_number;
        edge_vec.resize(edge_num);
        for (int i=0; i<edge_vec.size(); i++)
            edge_vec[i].resize(n);
        node_vec.resize(node_num);
        for (int i=0; i<node_vec.size(); i++)
            node_vec[i].resize(n);
        edge_vector_tmp.resize(edge_num);
        for (int i=0; i<edge_vector_tmp.size(); i++)
            edge_vector_tmp[i].resize(n);
        node_vector_tmp.resize(node_num);
        for (int i=0; i<node_vector_tmp.size(); i++)
            node_vector_tmp[i].resize(n);
        for (int i=0; i<edge_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                edge_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
        }
        for (int i=0; i<node_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                node_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
            norm(node_vec[i]);
        }

        SGD(network_name);
    }

private:
    int n;
    string network_name;
    double res;//loss function value
    double rate,margin;
    vector<int> src_list,dst_list,edge_list;
    vector<vector<double> > edge_vec,node_vec;
    vector<vector<double> > edge_vector_tmp,node_vector_tmp;
    bool L1_flag = 1;
    int generate_flag; // 生成正负样本的方法
    // 1. generate_flag = 0 ---> 一个正样本，随机对应一个负样本
    // 2. generate_flag = 1 ---> 按照拓扑结构生成正负样本: 采用思想：目的节点的邻居中不与源节点直接相连的点，都当做负样本（根据三角形原则。。。）
    int sample_number = 3; // 采样个数

    //add ignore fact
    //add time 2017-03-27
    set<string> ignore;

    int edge_num,node_num;
    map<string,int> edge2id,node2id;
    map<int,string> id2node,id2edge;

    network<std::string, double> layer;

    double norm(vector<double> &a)
    {
        double x = vec_len(a);
        if (x>1)
            for (int ii=0; ii<a.size(); ii++)
                a[ii]/=x;
        return 0;
    }
    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        while (res<0)
            res+=x;
        return res;
    }

    void SGD(string network_name)
    {
        res=0;
        int nbatches = 1; // Using GD instead of SGD

        int nepoch = 10000;
        int batchsize = (int) (src_list.size() / nbatches);
        int epoch;
        for (epoch = 0; epoch < nepoch; epoch++)
        {
            res=0; // 当前epoch下的loss function 值
            int total_ill_fact = 0; // 网络中总共生成的 ill 样本个数 --- 用在 CASE 1中

            for (int batch = 0; batch<nbatches; batch++) {
                edge_vector_tmp = edge_vec;
                node_vector_tmp = node_vec;
                for (int k = 0; k < batchsize; k++) {
                    /**
                     * 方法1： 随机生成一个正样本和一个负样本
                     */
                     switch (generate_flag)
                     {
                         case 0: {
                             int i = rand_max(src_list.size());
                             int j = rand_max(node_num);
                             if (rand() % 1000 < 500) {
                                 while (ok[make_pair(src_list[i], edge_list[i])].count(j) > 0)
                                     j = rand_max(node_num);
                                 train_fact(src_list[i], dst_list[i], edge_list[i], src_list[i], j, edge_list[i]);
                             } else {
                                 while (ok[make_pair(j, edge_list[i])].count(dst_list[i]) > 0)
                                     j = rand_max(node_num);
                                 train_fact(src_list[i], dst_list[i], edge_list[i], j, dst_list[i], edge_list[i]);
                             }
                             norm(edge_vector_tmp[edge_list[i]]);
                             norm(node_vector_tmp[src_list[i]]);
                             norm(node_vector_tmp[dst_list[i]]);
                             norm(node_vector_tmp[j]);
                             break;
                         } //case 0 end
                         case 1: {
                             /**
                              * 方法2：根据拓扑结构生成对应的 正样本和负样本
                             */
                             // 0. 设置sample的长度（因为不sample的话会引起 loss function 的震荡！！）

                             // 1. 生成正样本序号
//                             int i = rand_max(src_list.size());
                             int i = k;

                             // 2. 根据正样本序号生成对应负样本
                             string src = id2node[src_list[i]];
                             string dst = id2node[dst_list[i]];
                             string edge = id2edge[edge_list[i]];
                             set<string> src_neighbors = layer.neighbors(src);
                             set<string> dst_neighbors = layer.neighbors(dst);
                             //3. 负样本就是所有是dst的邻居，而不是src的邻居的节点

                             if (rand() % 1000 < 500) {
                                 //src_index
                                 int src_index = node2id[src];
                                 // 替换目的节点
                                 set<string> ill_dst_list;
                                 ill_dst_list = ill_fact_generate(src_neighbors, dst_neighbors);

                                 total_ill_fact += ill_dst_list.size();

                                 set<string> sample_ill_dst_fact = sample_list(ill_dst_list,sample_number);

//                                 int count = 0;
                                 for (auto it = sample_ill_dst_fact.begin(); it != sample_ill_dst_fact.end(); it++) {
                                     int ill_dst_id = node2id[*it];
                                     if (ill_dst_id != src_index) {
                                         int j = ill_dst_id;

                                         train_fact(src_list[i], dst_list[i], edge_list[i], src_list[i], j,
                                                    edge_list[i]);

                                         norm(edge_vector_tmp[edge_list[i]]);
                                         norm(node_vector_tmp[src_list[i]]);
                                         norm(node_vector_tmp[dst_list[i]]);
                                         norm(node_vector_tmp[j]);

//                                         if(count > 3){break;}
//                                         else{count++;}
                                     }
                                 }
                             } else {
                                 //dst_index
                                 int dst_index = node2id[dst];
                                 // 替换源节点
                                 set<string> ill_src_list;
                                 ill_src_list = ill_fact_generate(dst_neighbors, src_neighbors);

                                 total_ill_fact += ill_src_list.size();
                                 set<string> sample_ill_src_fact = sample_list(ill_src_list,sample_number);

//                                 int count = 0;
                                 for (auto it = sample_ill_src_fact.begin(); it != sample_ill_src_fact.end(); it++) {
                                     int ill_src_id = node2id[*it];
                                     if (ill_src_id != dst_index) {
                                         int j = ill_src_id;
                                         train_fact(src_list[i], dst_list[i], edge_list[i], j, dst_list[i],
                                                    edge_list[i]);

                                         norm(edge_vector_tmp[edge_list[i]]);
                                         norm(node_vector_tmp[src_list[i]]);
                                         norm(node_vector_tmp[dst_list[i]]);
                                         norm(node_vector_tmp[j]);

//                                         if(count > 3){break;}
//                                         else{count++;}
                                     }
                                 }
                             }
/*                             // 应刘博士要求，加上每次的res输出
                             // Add Time: 2017-03-26
                             if (i==0){
                                 cout << "\tcurrent_res: " << current_res << endl;
                             }*/
                             break;
                         }//case1 end
                     }//switch end
                }//for end
                edge_vec = edge_vector_tmp;
                node_vec = node_vector_tmp;

            }
//            cout<<"epoch:"<<epoch<<'\t'<<res<<"\ttotal_ill_fact: "<<total_ill_fact<<"\taverage_ill_fact: "<<total_ill_fact/src_list.size()<<endl;
            if (epoch % 10 == 0){
                cout<<"epoch:"<<epoch<<'\t'<<res;
                cout << "  ignore number: " << this->ignore.size() << endl;
            }

            FILE* f2 = fopen(("tmp/"+network_name+".node").c_str(),"w");
            FILE* f3 = fopen(("tmp/"+network_name+".edge").c_str(),"w");
            for (int i=0; i<edge_num; i++)
            {
                for (int ii=0; ii<n; ii++)
                    fprintf(f2,"%.6lf\t",edge_vec[i][ii]);
                fprintf(f2,"\n");
            }
            for (int i=0; i<node_num; i++)
            {
                for (int ii=0; ii<n; ii++)
                    fprintf(f3,"%.6lf\t",node_vec[i][ii]);
                fprintf(f3,"\n");
            }
            fclose(f2);
            fclose(f3);
            if (res < 0.0000001){break;}
        }

    }

    set<string> ill_fact_generate(set<string> src_neighbors, set<string> dst_neighbors)
    {
        //负样本就是所有是dst的邻居，而不是src的邻居的节点
        set<string> intersection;
        set_intersection(src_neighbors.begin(),src_neighbors.end(), dst_neighbors.begin(),dst_neighbors.end(),std::inserter(intersection,intersection.begin()));
        set<string> ill_results;
        set_difference(dst_neighbors.begin(), dst_neighbors.end(), intersection.begin(),intersection.end(),std::inserter(ill_results,ill_results.begin()));

        return ill_results;
    }
    double calc_sum(int src,int dst,int edge)
    {
        double sum=0;
        if (L1_flag)
            for (int ii=0; ii<n; ii++)
                sum+=fabs(node_vec[dst][ii]-node_vec[src][ii]-edge_vec[edge][ii]);
        else
            for (int ii=0; ii<n; ii++)
                sum+=sqr(node_vec[dst][ii]-node_vec[src][ii]-edge_vec[edge][ii]);
        return sum;
    }
    void gradient(int src_OK,int dst_OK,int edge_OK,int src_ill,int dst_ill,int edge_ill)
    {
        for (int ii=0; ii<n; ii++)
        {

            double x = 2*(node_vec[dst_OK][ii]-node_vec[src_OK][ii]-edge_vec[edge_OK][ii]);
            if (L1_flag) {
                if (x > 0) { x = 1; }
                else { x = -1; }
            }
            edge_vector_tmp[edge_OK][ii]-=-1*rate*x;
            node_vector_tmp[src_OK][ii]-=-1*rate*x;
            node_vector_tmp[dst_OK][ii]+=-1*rate*x;
            x = 2*(node_vec[dst_ill][ii]-node_vec[src_ill][ii]-edge_vec[edge_ill][ii]);
            if (L1_flag) {
                if (x > 0) { x = 1; }
                else { x = -1; }
            }
            edge_vector_tmp[edge_ill][ii]-=rate*x;
            node_vector_tmp[src_ill][ii]-=rate*x;
            node_vector_tmp[dst_ill][ii]+=rate*x;
        }
    }

    void train_fact(int src_OK, int dst_OK, int edge_OK, int src_ill, int dst_ill, int edge_ill)
    {
        double sum1 = calc_sum(src_OK,dst_OK,edge_OK);
        double sum2 = calc_sum(src_ill,dst_ill,edge_ill);

        // add 看fact 和 ill_fact
        // time:2017-03-27
        string current_infor = id2node[src_OK] + "---" + id2node[dst_OK] + "---" + id2node[src_ill] + "---" + id2node[dst_ill];
        if (this->ignore.find(current_infor) == this->ignore.end()){
            if (sum1+margin<=sum2) {
                this->ignore.insert(current_infor);
            }
            // end add

            if (sum1+margin>sum2)
            {
                res+=margin+sum1-sum2;
                gradient( src_OK, dst_OK, edge_OK, src_ill, dst_ill, edge_ill);
            }
        }


//        if (sum1+margin>sum2)
//        {
//            res+=margin+sum1-sum2;
//            gradient( src_OK, dst_OK, edge_OK, src_ill, dst_ill, edge_ill);
//        }
    }

    set<string> sample_list(set<string> list, int number){
        set<string> sample_results;

        if (number >= list.size()){ // 如果采样个数大于了原本的 ill 个数，直接返回就OK
            sample_results = list;
        } else {
            vector<string> tmp; // 将set转换成vector，好取下标。。。
            for (auto it = list.begin();it!=list.end();it++){
                tmp.push_back(*it);
            }

            bool continueFlag = true;
            int count = 0;
            while(continueFlag){
                int randomIdx = rand()%tmp.size();
                sample_results.insert(tmp[randomIdx]);
                if (sample_results.size() >= 3){
                    continueFlag = false;
                }
            }
        }
        return sample_results;
    }
};





int main(int argc,char**argv)
{
    /*
     * 简单形式
     * -network_name test -generate_flag 1
     *
     * 完整形式
     * -network_name test -generate_flag 1 -sample_number 3 -dim 100 -L1_flag 1 -rate 0.001 -margin 1 -show_info 1
     * */

    // 输入参数
    string network;
    int generate_flag; // 生成拓扑结构方式 随机生成负样本(=0) 、 根据三角形原理生成负样本(=1) 0

    // 默认参数
    int dimension = 100; // 维数 100
    float rate = 0.001; // 学习率 0.001
    double margin = 1; // Loss Function 中正样本和负样本之间的间隔 1
    int L1_flag = 1; // L1距离(=1)还是L2距离(=0) 1
    int show_info = 1; // 1
    int sample_number = 3; // 采样负样本的个数, 默认为3个

    // ----------------------------------------------------------------------------------
    int i;
    if ((i = ArgPos((char *)"-dim", argc, argv)) > 0) dimension = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-rate", argc, argv)) > 0) rate = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-network_name", argc, argv)) > 0) network = argv[i + 1];
    if ((i = ArgPos((char *)"-L1_flag", argc, argv)) > 0) L1_flag = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-show_info", argc, argv)) > 0) show_info = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-generate_flag", argc, argv)) > 0) generate_flag = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sample_number", argc, argv)) > 0) sample_number = atoi(argv[i + 1]);

    if (show_info == 1){
        cout<< "Set Information..." << endl;
        cout<<"dim = "<<dimension<<endl;
        cout<<"learning rate = "<<rate<<endl;
        cout<<"margin = "<<margin<<endl;
        cout<<"L1_Flag = "<<L1_flag<<endl;
        cout<<"generate_flag = "<<generate_flag<<endl;
        cout<<"network = "<<network<<endl;
        cout<<"sample_number = "<<sample_number<<endl;
    }
    // ----------------------------------------------------------------------------------
    string network_dir = network+"_tmp";

    // string node_id_path = "../"+network_dir+"/node_2_id.txt";
    // string edge_id_path = "../"+network_dir+"/edge_2_id.txt";
    // string fact_path = "../"+network_dir+"/src_dst_edge_fact.txt";
    string node_id_path = "./tmp/node_2_id.txt";
    string edge_id_path = "./tmp/edge_2_id.txt";
    string fact_path = "./tmp/src_dst_edge_fact.txt";

    Train train;
    train.prepare(node_id_path, edge_id_path, fact_path, (bool)L1_flag, generate_flag);
    train.run(dimension,rate,margin,network,sample_number);
}


