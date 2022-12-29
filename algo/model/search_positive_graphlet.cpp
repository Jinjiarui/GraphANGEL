#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string.h>
#include <vector>

const int N = 20000010;

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

struct list {
    int x, timestamp, type0, type1, next;
} T[N * 2];
int head[N], total;

int n, m, task;
int node_type0, node_type1;
std::vector<int> node_type;
std::vector<int> deg;
std::vector<std::pair<int, int> > vis;
std::vector<std::vector<std::pair<int, int> > > adj;
std::vector<triple<std::pair<int, int> > > triangles;
std::vector<quads<std::pair<int, int> > > quadrangles;

std::map<triple<int>, std::vector<triple<int> > > triangles_by_type;
std::map<quads<int>, std::vector<quads<int> > > quadrangles_by_type;

bool cmp(std::pair<int, int> x, std::pair<int, int> y) {
    if (deg[x.first] != deg[y.first]) {
        return deg[x.first] < deg[y.first];
    } else if (x.first != y.first) {
        return x.first < y.first;
    } else {
        return 0;
    }
}

void append(int u, int x, int timestamp, int type0 = 0, int type1 = 0) {
    T[total].x = x;
    T[total].timestamp = timestamp;
    T[total].type0 = type0;
    T[total].type1 = type1;
    T[total].next = head[u];
    head[u] = total++;
    if (total > N) {
        printf("out of memory");
        exit(0);
    }
}

void add_triangle_from_u(std::pair<int, int> u, std::pair<int, int> v, std::pair<int, int> w,
                         std::vector<triple<std::pair<int, int> > > &triangles) {
    if (node_type[u.first] == node_type0) {
        if (node_type[v.first] == node_type1) {
            triangles.push_back((triple<std::pair<int, int> >){u, w, v});
        }
        if (node_type[w.first] == node_type1) {
            triangles.push_back((triple<std::pair<int, int> >){u, v, w});
        }
    }
}

void add_quadrangle(std::pair<int, int> u, std::pair<int, int> v, std::pair<int, int> w, std::pair<int, int> x,
                    std::vector<quads<std::pair<int, int> > > &quadrangles) {
    if (node_type[u.first] == node_type0) {
        if (node_type[x.first] == node_type1) {
            // printf("%d 1\n", quadrangles.size());
            quadrangles.push_back((quads<std::pair<int, int> >){u, w, v, x});
            // printf("%d 2\n", quadrangles.size());
        }
    }
}

void add_quadrangle_from_uv(std::pair<int, int> u, std::pair<int, int> v, std::pair<int, int> w, std::pair<int, int> x,
                            std::vector<quads<std::pair<int, int> > > &quadrangles) {
    add_quadrangle(u, v, w, x, quadrangles);
    add_quadrangle(v, u, w, x, quadrangles);
}

void input() {
    scanf("%d%d%d", &n, &m, &task);
    scanf("%d%d", &node_type0, &node_type1);

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
        adj[u].push_back(std::make_pair(v, w));
        deg[u]++;
    }
}

void preprocess() {
    memset(head, -1, sizeof head);
    // remove duplicate edges & sort adj by degree
    for (int u = 0; u < n; u++) {
        std::sort(adj[u].begin(), adj[u].end(), cmp);
        adj[u].erase(unique(adj[u].begin(), adj[u].end()), adj[u].end());
    }
}

void search_triangle() {
    for (int u = 0; u < n; u++) {
        for (int i = 0; i < (int)adj[u].size(); i++) {
            int v = adj[u][i].first, vv = adj[u][i].second;
            vis[v] = std::make_pair(u, vv);
        }

        for (int i = 0; i < (int)adj[u].size(); i++) {
            int v = adj[u][i].first, vv = adj[u][i].second;
            if (deg[v] < deg[u] || deg[v] == deg[u] && v < u) {
                for (int j = 0; j < (int)adj[v].size(); j++) {
                    int w = adj[v][j].first, ww = adj[v][j].second;
                    if (deg[w] < deg[v] || deg[w] == deg[v] && w < v) {
                        // if (u, w) in E then (u, v, w) forms a triangle
                        if (vis[w].first == u) {
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
                                pw.second = vis[w].second;
                            }
                            add_triangle_from_u(pu, pv, pw, triangles);
                            add_triangle_from_u(pv, pw, pu, triangles);
                            add_triangle_from_u(pw, pu, pv, triangles);
                        }
                    } else {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }
    printf("%lld\n", (long long)triangles.size());
}

int cnt[N], cnt_time[N];

void search_quadrangle() {
    int timestamp = 0;
    long long total = 0;
    for (int u = 0; u < n; u++) {
        timestamp++;
        for (int i = 0; i < (int)adj[u].size(); i += 1) {
            int v = adj[u][i].first, vv = adj[u][i].second;
            if (deg[v] < deg[u] || deg[v] == deg[u] && v < u) {
                for (int j = 0; j < (int)adj[v].size(); j++) {
                    int w = adj[v][j].first, ww = adj[v][j].second;
                    if (deg[w] < deg[u] || deg[w] == deg[u] && w < u) {
                        if (cnt_time[w] != timestamp) {
                            cnt[w] = 0;
                            cnt_time[w] = timestamp;
                        } else {
                            total += cnt[w];
                            cnt[w]++;
                        }
                        // continue;
                        int last_k = -1;
                        for (int k = head[w]; ~k; k = T[k].next) {
                            int x = T[k].x, type0 = T[k].type0, type1 = T[k].type1;
                            if (T[k].timestamp != timestamp) {
                                if (~last_k) {
                                    T[last_k].next = -1;
                                }
                                break;
                            } else {
                                if (v < x) {
                                    std::pair<int, int> pu, pv, pw, px;
                                    pu.first = u;
                                    pv.first = v;
                                    pw.first = w;
                                    px.first = x;
                                    if (task != 1) {
                                        pu.second = node_type[u];
                                        pv.second = node_type[v];
                                        pw.second = node_type[w];
                                        pw.second = node_type[x];
                                    } else {
                                        pu.second = vv;
                                        pv.second = ww;
                                        pw.second = type1;
                                        px.second = type0;
                                    }
                                    // (u, v, w, x) forms a quadrangle
                                    add_quadrangle_from_uv(pu, pv, pw, px, quadrangles);
                                    add_quadrangle_from_uv(pv, pw, px, pu, quadrangles);
                                    add_quadrangle_from_uv(pw, px, pu, pv, quadrangles);
                                    add_quadrangle_from_uv(px, pu, pv, pw, quadrangles);
                                    // printf("%d %d %d %d\n", u, v, w, x);
                                }
                            }
                            last_k = k;
                        }
                        append(w, v, timestamp, vv, ww);
                    } else {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }
    printf("%lld\n", total);
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
    for (int i = 0; i < (int)quadrangles.size(); i++) {
        quads<std::pair<int, int> > &quadrangle = quadrangles[i];
        quads<int> quadrangle0 = (quads<int>){quadrangle.first.first, quadrangle.second.first, quadrangle.third.first,
                                              quadrangle.forth.first};
        quads<int> quadrangle_type;
        if (task != 1) {
            quadrangle_type = (quads<int>){node_type[quadrangle.first.first], node_type[quadrangle.second.first],
                                           node_type[quadrangle.third.first], node_type[quadrangle.forth.first]};
        } else {
            quadrangle_type = (quads<int>){quadrangle.first.second, quadrangle.second.second, quadrangle.third.second,
                                           quadrangle.forth.second};
        }

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
    search_quadrangle();
    arrange_by_type();
    output_triangles(output_triangles_filename);
    output_quadrangles(output_quadrangles_filename);
}

int main(int argc, char *argv[]) {
    setbuf(stdout, NULL);
    std::string dataset = std::string(argv[1]);

    std::string input_filename = "output/" + dataset + "_graph.txt";
    std::string output_triangles_filename = "output/" + dataset + "_triangles.txt";
    std::string output_quadrangles_filename = "output/" + dataset + "_quadrangles.txt";

    freopen(input_filename.c_str(), "r", stdin);

    process(output_triangles_filename, output_quadrangles_filename);

    return 0;
}
