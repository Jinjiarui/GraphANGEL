#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string.h>
#include <vector>

const int N = 7000010, M = 10000010;
const int sample_count = 200000;

template <class T> struct triple {
    T first, second, third;

    bool operator<(const triple<T> &rhs) const {
        if (first != rhs.first) {
            return first < rhs.first;
        } else if (second != rhs.second) {
            return second < rhs.second;
        } else {
            return third < rhs.third;
        }
    }
};

template <class T> struct quads {
    T first, second, third, forth;

    bool operator<(const quads<T> &rhs) const {
        if (first != rhs.first) {
            return first < rhs.first;
        } else if (second != rhs.second) {
            return second < rhs.second;
        } else if (third != rhs.third) {
            return third < rhs.third;
        } else {
            return forth < rhs.forth;
        }
    }
};

int n, m, task;
int node_type0, node_type1;
std::vector<int> node_type;
std::vector<int> deg;
std::vector<std::pair<int, int> > vis;
std::vector<std::vector<std::pair<int, int> > > adj;
std::vector<triple<int> > edges;
std::set<std::pair<int, int> > edges_set;
std::vector<triple<std::pair<int, int> > > triangles;
std::set<quads<std::pair<int, int> > > quadrangles;

long long c[M], sc[M], total_c;

std::map<triple<int>, std::vector<triple<int> > > triangles_by_type;
std::map<quads<int>, std::vector<quads<int> > > quadrangles_by_type;

unsigned int rand32() {
    return rand() << 16 ^ rand();
}

unsigned long long rand64() {
    return (unsigned long long)rand32() << 32 ^ rand32();
}

bool cmp(std::pair<int, int> x, std::pair<int, int> y) {
    if (deg[x.first] != deg[y.first]) {
        return deg[x.first] < deg[y.first];
    } else if (x.first != y.first) {
        return x.first < y.first;
    } else {
        return 0;
    }
}

// void append(int u, int x, int timestamp) {
//     T[total].x = x;
//     T[total].timestamp = timestamp;
//     T[total].next = head[u];
//     head[u] = total++;
//     if (total > N) {
//         printf("out of mem");
//         exit(0);
//     }
// }

void add_triangle(std::pair<int, int> u, std::pair<int, int> v, std::pair<int, int> w,
                  std::vector<triple<std::pair<int, int> > > &triangles) {
    assert(node_type[u.first] == node_type0);
    assert(node_type[w.first] == node_type1);
    triangles.push_back((triple<std::pair<int, int> >){u, v, w});
}

void add_quadrangle(std::pair<int, int> u, std::pair<int, int> v, std::pair<int, int> w, std::pair<int, int> x,
                    std::set<quads<std::pair<int, int> > > &quadrangles) {
    if (node_type[u.first] == node_type0) {
        if (node_type[x.first] == node_type1) {
            quadrangles.insert((quads<std::pair<int, int> >){u, v, w, x});
        }
    }
}

void input() {
    scanf("%d%d%d", &n, &m, &task);
    scanf("%d%d", &node_type0, &node_type1);

    assert(N > n && M > m);

    node_type.resize(n);
    deg.resize(n);
    vis.resize(n);
    adj.resize(n);

    for (int i = 0; i < n; i++) {
        scanf("%d", &node_type[i]);
        deg[i] = 0;
        vis[i] = std::make_pair(-1, -1);
    }

    for (int i = 0; i < m; i++) {
        int u, v, w;
        scanf("%d%d", &u, &v);
        if (task == 1) {
            scanf("%d", &w);
        } else {
            w = 0;
        }
        if (u == v) {
            printf("%d\n", u);
        }
        assert(u != v);
        edges.push_back((triple<int>){u, v, w});
        edges_set.insert(std::make_pair(u, v));
        adj[u].push_back(std::make_pair(v, w));
        deg[u]++;
    }
}

void preprocess() {
    // remove duplicate edges & sort adj by degree
    for (int u = 0; u < n; u++) {
        std::sort(adj[u].begin(), adj[u].end(), cmp);
        int a = (adj[u].size());
        adj[u].erase(unique(adj[u].begin(), adj[u].end()), adj[u].end());
    }
}

void search_triangle() {
    for (int u = 0; u < n; u++) {
        if (node_type[u] != node_type0) {
            continue;
        }
        for (int i = 0; i < (int)adj[u].size(); i++) {
            int v = adj[u][i].first, vv = adj[u][i].second;
            vis[v] = std::make_pair(u, vv);
        }
        for (int i = 0; i < (int)adj[u].size(); i++) {
            // if (i > 10) {
            //     break;
            // }
            int v = adj[u][i].first, vv = adj[u][i].second;
            for (int j = 0; j < (int)adj[v].size(); j++) {
                // if (j > 10) {
                //     break;
                // }
                int w = adj[v][j].first, ww = adj[v][j].second;
                if (node_type[w] != node_type1) {
                    continue;
                }
                if (vis[w].first != u) {
                    std::pair<int, int> pu, pv, pw;
                    pu.first = u;
                    pv.first = v;
                    pw.first = w;
                    if (task != 1) {
                        pu.second = node_type[u];
                        pv.second = node_type[v];
                        pw.second = node_type[w];
                    } else {
                        pu.second = vv;
                        pv.second = ww;
                        pw.second = 0;
                    }

                    add_triangle(pu, pv, pw, triangles);
                }
            }
        }
    }
    printf("%lu\n", triangles.size());
}

int cnt[N], cnt_time[N];

void sample_quadrangle() {
    for (int i = 0; i < m; i++) {
        int u = edges[i].first, v = edges[i].second;
        c[i] = (long long)(deg[u] - 1) * (deg[v] - 1);
        sc[i] = c[i];
        if (i > 0) {
            sc[i] += sc[i - 1];
        }
    }
    total_c = sc[m - 1];
    while (quadrangles.size() < sample_count) {
        long long sample_w = rand64() % total_c;
        int index = std::lower_bound(sc, sc + m, sample_w) - sc;

        int v = edges[index].first, w = edges[index].second, t = edges[index].third;
        int v_idx = rand32() % adj[v].size(), w_idx = rand32() % adj[w].size();
        std::pair<int, int> pu = adj[v][v_idx];
        std::pair<int, int> px = adj[w][w_idx];
        int u = pu.first, x = px.first;
        if (u == w || v == x) {
            continue;
        }
        if (edges_set.count(std::make_pair(u, x))) {
            continue;
        }

        std::pair<int, int> pv = std::make_pair(v, t), pw = std::make_pair(w, px.second);
        px = std::make_pair(x, 0);

        // if (quadrangles.size() % 1000 == 0) {
        //     printf("*** %d\n", node_type0, node_type1);
        //     printf("%d %d\n", node_type[pu.first], node_type[px.first]);
        //     printf("%lu\n", quadrangles.size());
        // }

        add_quadrangle(pu, pv, pw, px, quadrangles);
    }
}

void arrange_by_type() {
    for (int i = 0; i < (int)triangles.size(); i++) {
        triple<std::pair<int, int> > &triangle = triangles[i];

        triple<int> triangle0 = (triple<int>){triangle.first.first, triangle.second.first, triangle.third.first};

        triple<int> triangle_type;
        if (task != 1) {
            triangle_type = (triple<int>){node_type[triangle.first.first], node_type[triangle.second.first],
                                          node_type[triangle.third.first]};
        } else {
            triangle_type = (triple<int>){triangle.first.second, triangle.second.second, triangle.third.second};
        }

        triangles_by_type[triangle_type].push_back(triangle0);
    }
    // for (int i = 0; i < (int)quadrangles.size(); i++) {
    //     quads<std::pair<int, int> > &quadrangle = quadrangles[i];
    //     quads<int> quadrangle0 = (quads<int>){quadrangle.first.first, quadrangle.second.first,
    //     quadrangle.third.first,
    //                                           quadrangle.forth.first};
    //     quads<int> quadrangle_type;
    //     if (node_type0 != node_type1) {
    //         quadrangle_type = (quads<int>){node_type[quadrangle.first.first], node_type[quadrangle.second.first],
    //                                        node_type[quadrangle.third.first], node_type[quadrangle.forth.first]};
    //     } else {
    //         quadrangle_type = (quads<int>){quadrangle.first.second, quadrangle.second.second,
    //         quadrangle.third.second, quadrangle.forth.second};
    //     }

    //     quadrangles_by_type[quadrangle_type].push_back(quadrangle0);
    // }

    for (std::set<quads<std::pair<int, int> > >::iterator it = quadrangles.begin(); it != quadrangles.end(); it++) {
        // for (int i = 0; i < (int)quadrangles.size(); i++) {
        quads<std::pair<int, int> > quadrangle = *it;
        // quads &quadrangle = quadrangles[i];
        quads<int> quadrangle0 = (quads<int>){quadrangle.first.first, quadrangle.second.first, quadrangle.third.first,
                                              quadrangle.forth.first};
        quads<int> quadrangle_type =
            (quads<int>){node_type[quadrangle.first.second], node_type[quadrangle.second.second],
                         node_type[quadrangle.third.second], node_type[quadrangle.forth.second]};

        quadrangles_by_type[quadrangle_type].push_back(quadrangle0);
    }
}

void output_triangles(std::string filename) {
    freopen(filename.c_str(), "w", stdout);
    printf("%d\n", (int)triangles_by_type.size());
    for (std::map<triple<int>, std::vector<triple<int> > >::iterator it = triangles_by_type.begin();
         it != triangles_by_type.end(); it++) {
        std::vector<triple<int> > triangles = it->second;
        printf("%d %d %d %d\n", node_type[triangles[0].first], node_type[triangles[0].second],
               node_type[triangles[0].third], (int)triangles.size());
        for (int i = 0; i < (int)triangles.size(); i++) {
            printf("%d %d %d\n", triangles[i].first, triangles[i].second, triangles[i].third);
        }
    }
}

void output_quadrangles(std::string filename) {
    freopen(filename.c_str(), "w", stdout);
    printf("%d\n", (int)quadrangles_by_type.size());
    for (std::map<quads<int>, std::vector<quads<int> > >::iterator it = quadrangles_by_type.begin();
         it != quadrangles_by_type.end(); it++) {
        std::vector<quads<int> > quadrangles = it->second;
        printf("%d %d %d %d %d\n", node_type[quadrangles[0].first], node_type[quadrangles[0].second],
               node_type[quadrangles[0].third], node_type[quadrangles[0].forth], (int)quadrangles.size());
        for (int i = 0; i < (int)quadrangles.size(); i++) {
            printf("%d %d %d %d\n", quadrangles[i].first, quadrangles[i].second, quadrangles[i].third,
                   quadrangles[i].forth);
        }
    }
}

void process(std::string output_triangles_filename, std::string output_quadrangles_filename) {
    input();
    preprocess();
    search_triangle();
    sample_quadrangle();
    arrange_by_type();
    output_triangles(output_triangles_filename);
    output_quadrangles(output_quadrangles_filename);
}

int main(int argc, char *argv[]) {
    // srand(time(0));
    std::string dataset = std::string(argv[1]);

    std::string input_filename = "../data/" + dataset + "/cache/" + dataset + "_graph.txt";
    std::string output_triangles_filename = "../data/" + dataset + "/cache/" + dataset + "_neg_triangles.txt";
    std::string output_quadrangles_filename = "../data/" + dataset + "/cache/" + dataset + "_neg_quadrangles.txt";

    std::cout << input_filename << std::endl;
    std::cout << output_triangles_filename << std::endl;
    std::cout << output_quadrangles_filename << std::endl;

    freopen(input_filename.c_str(), "r", stdin);

    process(output_triangles_filename, output_quadrangles_filename);

    return 0;
}