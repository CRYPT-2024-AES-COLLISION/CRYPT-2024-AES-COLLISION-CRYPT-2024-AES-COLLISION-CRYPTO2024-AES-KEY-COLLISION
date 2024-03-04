import sys
from collections import defaultdict
sys.setrecursionlimit(10**6)
par = {}
# 繋がっているinboundノードを調べるためのデータ構造（Union-Find）
def root(x):
    if par[x] < 0:
        return x
    else:
        par[x] = root(par[x])
        return par[x]

def union(x, y):
    xx = root(x)
    yy = root(y)
    if xx == yy:
        return
    else:
        par[xx] += par[yy]
        par[yy] = xx

def same(x, y):
    xx = root(x)
    yy = root(y)
    return xx == yy



# グループ分けのための深さ優先探索
def dfs(in_dir,node,group):
    forward = 0b1111 ^ (1<<in_dir)
    # print(in_dir,node,bin(forward))

    already_visited = dic[node[0]][node[1]][node[2]].activate(in_dir,node)

    dic[node[0]][node[1]][node[2]].group_update(group,node)
    

    for i in range(4):
        if (forward >> i) & 1:
            ret = dic[node[0]][node[1]][node[2]].forward(i)
            
            for d in ret:
                # print(node, "->", d)
                if d[0] == "": continue
                # 次に渡すグループの値を計算
                # (どのinboundの自由度によって値が作られているかを調べる)
                if node[0] == "enc_mc":
                    node = (node[0],node[1],d[2])
                if node[0] == "enc_mc":
                    for j in range(4):
                        union(node2idx[(node[0],node[1],0)], node2idx[(node[0],node[1],j)])
                # if same(node2idx[("enc_prob",6,0)], node2idx[("enc_xor",5,1)]):
                #     print("ok")
                g = 0
                g1 = 0
                g2 = 0
                x = node2idx[node]
                y = node2idx[d]
                for in_d in inbound_node:
                    if same(x, node2idx[in_d]):
                        g |= 1 << inbound_idx[in_d]
                        g1 |= 1 << inbound_idx[in_d]
                for in_d in inbound_node:
                    if same(y, node2idx[in_d]):
                        g |= 1 << inbound_idx[in_d]
                        g2 |= 1 << inbound_idx[in_d]
                
                # print(bin(g1),"+",bin(g2),node,"->",d)

                # print(bin(g))
                # ----- dof tree の作成 -----
                if "xor" in node[0] and "xor" not in d[0]:
                    child = dic[node[0]][node[1]][node[2]].con_node
                    parent = g
                    if child[0] != g and child[1] != g:
                        G[parent].add(child[0])
                        G[parent].add(child[1])
                        # G[child[0]].add(parent)
                        # G[child[1]].add(parent)
                if "xor" not in node[0] and "xor" in d[0]:
                    if len(dic[d[0]][d[1]][d[2]].con_node) == 1:
                        dic[d[0]][d[1]][d[2]].con_node[0] = g2
                    dic[d[0]][d[1]][d[2]].con_node.append(g1)
                    # print(node, g1, g2)
                if "mc" not in node[0] and "mc" in d[0]:
                    dic[d[0]][d[1]][d[2]].con_node.append(g1)
                if "mc" in node[0] and "mc" not in d[0]:
                    mixed_group = 0
                    for nodes in dic[node[0]][node[1]][node[2]].con_node:
                        mixed_group |= nodes
                    
                    connected_group = g

                    if mixed_group | connected_group != g:
                        G[mixed_group | connected_group].add(g)

                # ----------------------------
                # ノードの連結
                union(node2idx[node], node2idx[d])

                dfs(i, d, g)
    

class Node:
    def __init__(self):
        self.dir = [("",-100,-100),("",-100,100),("",-100,100),("",-100,100)]
        self.active = 0
        self.group = 0
        self.dof_group = 0
        self.prob = 0
    
    def forward(self,idx):
        # print(self.dir[idx],idx)
        if self.dir[idx][0] == "":
            return []
        else:
            return [self.dir[idx]]

    def activate(self, in_dir, node):
        self.active = True
        return True
    
    def group_merge(self,g):
        return g
    
    def group_update(self,g,node):
        self.group = g

class MC:
    def __init__(self):
        self.active = [0,0,0]
        self.dir = [[("",-100,-100),("",-100,100),("",-100,100),("",-100,100)], [("",-100,-100),("",-100,100),("",-100,100),("",-100,100)], [("",-100,-100),("",-100,100),("",-100,100),("",-100,100)]]
        self.group = [0,0,0,0]
        self.con_node = []
    
    def forward(self,idx):
        if idx != 0 and idx != 2:
            return []
        if self.active[0] == 0b1111:
            return self.dir[2]
        elif self.active[2] == 0b1111:
            return self.dir[0]
        else:
            return []
    
    def activate(self, in_dir, node):
        self.active[in_dir] |= 1<<node[2]
        return False
    
    def group_merge(self,g):
        return g
    
    def group_update(self,g,node):
        self.group[0] |= g
        self.group[1] |= g
        self.group[2] |= g
        self.group[3] |= g

            
class XOR:
    def __init__(self):
        self.active = 0
        self.dir = [("",-100,-100),("",-100,100),("",-100,100),("",-100,100)]
        self.group = 0
        self.dof_group = 0
        self.con_node = []

    def activate(self, in_dir, node):
        if self.active & 1<<in_dir:
            return True
        self.active |= 1<<in_dir
        # if "key_xor" == node[0]:
            # print(f"------------------- activate {node}:{in_dir} ---------------------------")
            # print(bin(self.active))
        return False
    
    def forward(self,idx):
        if self.active & (1<<idx) == 0 and bin(self.active).count("1") == 3:
            nxt = self.active ^ 0b1111
            bit = 1
            cnt = 0
            while nxt & (bit<<cnt) == 0 and cnt < 4:
                cnt += 1
            self.activate(cnt, ("",-1,-1))
            return [self.dir[cnt]]
        else:
            return []

    
    def group_merge(self,g):
        return g | self.group
    
    def group_update(self,g, node):
        self.group = g



round = 9
G = defaultdict(set)


enc_xor = [[XOR(),XOR(),XOR(),XOR()] for _ in range(round+1)]
key_xor = [[XOR(),XOR(),XOR(),XOR(), XOR(),XOR(),XOR(),XOR()] for _ in range((round+1)//2)]

enc_prob = [[Node(),Node(),Node(),Node()] for _ in range(round+2)]
key_prob = [[Node(),Node(),Node(),Node(), Node(),Node(),Node(),Node()] for _ in range((round+2)//2)]

enc_mc = [[MC()]*4 for _ in range(round+1)]

dic = {
    "enc_xor": enc_xor,
    "key_xor": key_xor,
    "enc_prob": enc_prob,
    "key_prob": key_prob,
    "enc_mc": enc_mc,
}

# どのノード同士が繋がっているかを手動で設定（グラフの構築）
# ----- enc_prob, MC -----
for i in range(4):
    dic["enc_prob"][0][i].dir[0] = ("enc_xor", 0,i)

for i in range(1, round+2):
    for j in range(4):
        dic["enc_prob"][i][j].dir[2] = ("enc_mc",i,j)
        dic["enc_prob"][i][j].dir[1] = ("enc_xor",i-1,j)

for i in range(4):
    dic["enc_prob"][round][i].dir[0] = ("enc_xor",round,i)

for i in range(4):
    dic["enc_prob"][round+1][i].dir[2] = ("",-100,-100)


# ----- enc_mc -----
for i in range(round+1):
    for j in range(4):
        dic["enc_mc"][i][0].dir[2][j] = ("enc_prob",i,j)
        dic["enc_mc"][i][0].dir[0][j] = ("enc_xor",i,j)

# ----- key_prob -----
for i in range((round+2)//2):
    for j in range(8):
        dic["key_prob"][i][j].dir[0] = ("key_xor", i,j)
        dic["key_prob"][i][j].dir[1] = ("key_xor",i-1,j)
        dic["key_prob"][i][j].dir[2] = ("key_xor",i-1,j+1)
        dic["key_prob"][i][j].dir[3] = ("enc_xor",i*2+j//4,j%4)



# 最初のラウンドの上方向
for j in range(8):
    dic["key_prob"][0][j].dir[1] = ("",-100,-100)

# 最初のラウンドの2方向
for j in range(7):
    dic["key_prob"][0][j].dir[2] = ("",-100,-100)

#　最後のラウンドの下方向
for j in range(8):
    dic["key_prob"][(round+2)//2-1][j].dir[0] = ("",-100,-100)

# 全ラウンドの鍵スケジュールの一番右
for i in range((round+2)//2):
    dic["key_prob"][i][7].dir[2] = ("key_xor",i,0)

# 最後のラウンドの一番右
dic["key_prob"][(round+2)//2-1][7].dir[2] = ("",-100,-100)

if round % 2 == 0:
    for j in range(4,8):
        dic["key_prob"][(round+2)//2-1][j].dir[3] = ("",-100,-100)

# ----- enc_xor -----
for i in range(round+1):
    for j in range(4):
        dic["enc_xor"][i][j].dir[0] = ("enc_mc",i,j)
        dic["enc_xor"][i][j].dir[1] = ("enc_prob",i+1,j)
        dic["enc_xor"][i][j].dir[3] = ("key_prob",i//2,(i%2*4)+j)
        dic["enc_xor"][i][j].active = 0b0100

# ----- key_xor -----
for i in range((round+1)//2):
    for j in range(8):
        dic["key_xor"][i][j].dir[0] = ("key_prob", i,j)
        dic["key_xor"][i][j].dir[1] = ("key_prob", i+1,j)
        dic["key_xor"][i][j].dir[2] = ("key_prob", i+1,j-1)
        dic["key_xor"][i][j].active = 0b1000

for i in range((round+1)//2):
    dic["key_xor"][i][0].dir[2] = ("key_prob", i,7)

# for j in range(8):
#     dic["key_xor"][(round+1)//2-1][j].dir[1] = ("", -100, -100)
#     dic["key_xor"][(round+1)//2-1][j].dir[2] = ("", -100, -100)

# ---------------------------------------------- ファイル読み込み ----------------------------------------------
key_cnt = 0
enc_cnt = 1
with open("log_done/AES256_relkey_collision/AES256_relkey_col_double_R9.txt", "r") as f:
    for line in f:
        if "dp" in line:
            if "key" in line:
                if key_cnt % 2 == 0:
                    six = line.count("□■")
                    seven = line.count("■■")
                    dic["key_prob"][key_cnt//2][3].prob = six*6+seven*7
                else:
                    six = line.count("□■")
                    seven = line.count("■■")
                    dic["key_prob"][key_cnt//2][7].prob = six*6+seven*7
                key_cnt += 1
            if "enc" in line:
                line = line.strip("\n")
                prob_log = line.split(" ")[1:]
                prob_list = []
                for i in range(16):
                    if prob_log[i] == "□■":
                        prob_list.append(6)
                    if prob_log[i] == "■■":
                        prob_list.append(7)
                    if prob_log[i] == "□□":
                        prob_list.append(0)
                for i in range(4):
                    dic["enc_prob"][enc_cnt][i].prob = sum(prob_list[i*4:(i+1)*4])
                enc_cnt += 1
                


import math
node2idx = {}
idx2node = {}

for k in dic:
    for i in range(len(dic[k])):
        for j in range(len(dic[k][i])):
            par[(k,i,j)] = -1
            node2idx[(k,i,j)] = len(node2idx)
            idx2node[len(idx2node)] = (k,i,j)

par = [-1]*len(node2idx)
done = set()

# inbooundするノードを決め打ち
inbound_node = [("enc_prob",0,j) for j in range(4)] + [("enc_prob",1,j) for j in range(3,-1,-1)] + [("enc_prob",2,j) for j in range(3,-1,-1)]
inbound_idx = dict()

for d in [("enc_prob", 0, j) for j in range(4)]:
    inbound_idx[d] = 0
for i, d in enumerate([("enc_prob",1,j) for j in range(3,-1,-1)] + [("enc_prob",2,j) for j in range(3,-1,-1)]):
    inbound_idx[d] = i+1

# 探索の実行
for i, d in enumerate(inbound_node):
    if d[1] == 0:
        dfs(5, d, 1<<0)
    else:
        dfs(5,d,1<<(i-3))

print(G)
print(dic["enc_xor"][5][0].con_node)
# print("----- enc_prob 確率-----")
# for d in dic["enc_prob"]:
#     for dd in d:
#         print(format(dd.group, "012b").replace("1","■").replace("0","□"), end=", ")

#     print()

print("----- enc_prob (group) -----")
for d in dic["enc_prob"]:
    for dd in d:
        print(format(dd.group, "012b").replace("1","■").replace("0","□"), end=", ")

    print()

print("----- key_prob (group) -----")
for d in dic["key_prob"]:
    for i, dd in enumerate(d):
        print(format(dd.group, "012b").replace("1","■").replace("0","□"), end=", ")

    print()


print("----- enc_mc -----")
for d in dic["enc_mc"]:
    for dd in d[0].group:
        print(format(dd, "012b").replace("1","■").replace("0","□"), end=", ")
    print()

# print("----- enc_xor -----")
# for d in dic["enc_xor"]:
#     for dd in d:
#         print(format(dd.active, "04b").replace("1","■").replace("0","□"), end=", ")
#     print()

# print("----- key_xor -----")
# for d in dic["key_xor"]:
#     for dd in d:
#         print(format(dd.active, "04b").replace("1","■").replace("0","□"), end=", ")
#     print()


# ----------- グループごと確率を計算する --------------------------
prob_dic = {}

for d in dic["enc_prob"]:
    for i in range(len(d)):
        prob_dic.setdefault(d[i].group, 0)
        prob_dic[d[i].group] += d[i].prob

for d in dic["key_prob"]:
    for i in range(len(d)):
        prob_dic.setdefault(d[i].group, 0)
        prob_dic[d[i].group] += d[i].prob

for d in dic["enc_mc"]:
    if d[0].group[0] not in prob_dic:
        prob_dic[d[0].group[0]] = 0

print(prob_dic)

print(G)
# ------------------- 計算量評価 & 自由度の計算 by MILP --------------------------

from sage.all import *

p = MixedIntegerLinearProgram(maximization=False)
v = p.new_variable(real=True, nonnegative=True)

in_cnt = 1
out_cnt = 1

variables = set()

def make_milp_constraint(s, parent, dof_out):
    nxt = list(G[s])
    if len(nxt) == 0:
        return
    d1, d2 = nxt[0], nxt[1]
    global in_cnt, out_cnt
    # print(s)
    print(d1,d2)
    if bin(d1).count("1") == 1:
        if d1 == 1:
            dof = 0
        else:
            dof = 32 - prob_dic[d1]

        val1_name = f"dof_in_{in_cnt}"
        val1 = v[val1_name]
        variables.add(f"dof_in_{in_cnt}")
        p.add_constraint(val1 <= dof)
        # print(f"dof_in_{d1} <= {dof}")
        in_cnt += 1
    else:
        val1_name = f"dof_out_{out_cnt}"
        val1 = v[val1_name]
        variables.add(f"dof_out_{out_cnt}")
        out_cnt += 1
        make_milp_constraint(d1, s, val1)
    
    if bin(d2).count("1") == 1:
        if d2 == 1:
            dof = 0
        else:
            dof = 32 - prob_dic[d2]
        val2_name = f"dof_in_{in_cnt}"
        val2 = v[val2_name]
        variables.add(f"dof_in_{in_cnt}")
        p.add_constraint(val2 <= dof)
        # print(f"dof_in_{d2} <= {dof}")
        in_cnt += 1
    else:
        val2_name = f"dof_out_{out_cnt}"
        val2 = v[val2_name]
        variables.add(f"dof_out_{out_cnt}")
        out_cnt += 1
        make_milp_constraint(d2, s, val2)
    
    if bin(s).count("1") > 1 and s > 0:
        p.add_constraint(val1 + val2 - prob_dic[s] == dof_out)
        # print(f"{val1_name} + {val2_name} - {prob_dic[s]} == dof_out")
        p.add_constraint(val1 + val2 <= v["max_dof"])

# 必要な条件
# dofの最大値の最小化 (dof1 + dof2 <= DOF_max)
# 自由度の計算 (dof1 == dof3 + dof4 + prob)
# inbound条件 (dof_inbound <= dof)

make_milp_constraint(max(prob_dic),-1 , v["dof_root"])

# p.add_constraint(v["prob_root"] == 32)
# p.add_constraint(v["dof_root"] == v["dof1"] + v["dof2"])
# p.add_constraint(v["dof_root"] >= 0)
# p.add_constraint(v["dof1"] + v["dof2"] <= 63)
# p.add_constraint(v["prob_root"] <= v["dof_root"])
# p.add_constraint(v["dof2"] <= 30)
# p.add_constraint(v["dof1"] == v["dof3"] + v["dof4"] - v["prob1"])
# p.add_constraint(v["prob1"] == 20)
# p.add_constraint(v["prob1"] <= v["dof3"] + v["dof4"])
# p.add_constraint(v["dof3"] + v["dof4"] <= 63)
# p.add_constraint(v["dof3"] <= 60)
# p.add_constraint(v["dof4"] <= 60)

p.set_objective(v["max_dof"])

round(p.solve())
print("Time Complexity ->", p.get_values(v["max_dof"]))
