#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <omp.h>

struct triple
{
    int first, second, third;

    bool operator<(const triple &rhs) const
    {
        if (first != rhs.first)
        {
            return first < rhs.first;
        }
        else if (second != rhs.second)
        {
            return second < rhs.second;
        }
        else
        {
            return third < rhs.third;
        }
    }
};

int n, m;
int node_type0, node_type1;
std::vector<int> node_type;
std::vector<int> deg, vis;
std::vector<std::vector<int>> adj;
std::vector<triple> triangles;

std::map<triple, std::vector<triple>> triangles_by_type;

bool cmp(int x, int y)
{
    if (deg[x] != deg[y])
    {
        return deg[x] < deg[y];
    }
    else if (x != y)
    {
        return x < y;
    }
    else
    {
        return 0;
        // exit(0);
    }
    // return deg[x] != deg[y] ? deg[x] < deg[y] : x < y;
}

void add_triangle_from_u(int u, int v, int w, std::vector<triple> &triangles)
{
    // std::cerr << node_type[u] << " " << node_type0 << std::endl;
    if (node_type[u] == node_type0)
    {
        // std::cerr << "here" << std::endl;
        if (node_type[v] == node_type1)
        {
            // std::cerr << "really add" << std::endl;
            triangles.push_back((triple){u, w, v});
        }
        if (node_type[w] == node_type1)
        {
            // std::cerr << "really add" << std::endl;
            triangles.push_back((triple){u, v, w});
        }
    }
}

void input()
{
    int t;
    scanf("%d%d%d", &n, &m, &t);
    scanf("%d%d", &node_type0, &node_type1);

    node_type.resize(n);
    deg.resize(n);
    vis.resize(n);
    adj.resize(n);

    for (int i = 0; i < n; i++)
    {
        scanf("%d", &node_type[i]);
        deg[i] = 0;
        vis[i] = -1;
    }

    for (int i = 0; i < m; i++)
    {
        int u, v;
        scanf("%d%d", &u, &v);
        // u--;
        // v--;
        adj[u].push_back(v);
        // adj[v].push_back(u);
        deg[u]++;
    }
}

void triangle_search()
{
    // remove duplicate edges & sort adj by degree
    for (int u = 0; u < n; u++)
    {
        std::sort(adj[u].begin(), adj[u].end(), cmp);
        int a = (adj[u].size());
        adj[u].erase(unique(adj[u].begin(), adj[u].end()), adj[u].end());
    }

    // int count = 0;
    // for (int u = 0; u < n; u++) {
    //     if (node_type[u] != 2) continue;
    //     for (int i = 0; i < (int) adj[u].size(); i++) {
    //         int v = adj[u][i];
    //         vis[v] = u;
    //     }
    //     for (int i = 0; i < (int) adj[u].size(); i++) {
    //         int v = adj[u][i];
    //         if (node_type[v] != 1) continue;
    //         for (int j = 0; j < (int) adj[v].size(); j++) {
    //             int w = adj[v][j];
    //             if (node_type[w] != 0) continue;
    //             if (vis[w] == u) {
    //                 count++;
    //             }
    //         }
    //     }
    // }

size_t omp_num_threads = omp_get_max_threads();

    std::cerr << "num of threads: " << omp_num_threads << std::endl;
    std::vector<std::vector<triple>> triangles_thread(omp_num_threads);
    // for (int i = 0 ; i < omp_get_max_threads() ; ++i)
        // triangles_thread[i].reserve(n);

#pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < n; u++)
    {
        std::size_t rank = omp_get_thread_num();
        std::vector<int> vis(n, -1);
        for (int i = 0; i < (int)adj[u].size(); i++)
        {
            int v = adj[u][i];
            vis[v] = u;
        }

        for (int i = 0; i < (int)adj[u].size(); i++)
        {
            int v = adj[u][i];
            if (deg[v] < deg[u] || deg[v] == deg[u] && v < u)
            {
                for (int j = 0; j < (int)adj[v].size(); j++)
                {
                    int w = adj[v][j];
                    if (deg[w] < deg[v] || deg[w] == deg[v] && w < v)
                    {
                        // if (u, w) in E then (u, v, w) forms a triangle
                        if (vis[w] == u)
                        {
                            add_triangle_from_u(u, v, w, triangles_thread[rank]);
                            add_triangle_from_u(v, w, u, triangles_thread[rank]);
                            add_triangle_from_u(w, u, v, triangles_thread[rank]);
                            // std::cerr <<"add: " << triangles_thread[rank].size() << std::endl;
                        }
                    }
                    else
                    {
                        break;
                    }
                }
            }
            else
            {
                break;
            }
        }
    }

    for (int i = 0; i < omp_num_threads; ++i)
    {
        std::cerr << triangles_thread[i].size() << std::endl;
        triangles.insert(triangles.end(), triangles_thread[i].begin(), triangles_thread[i].end());
    }
}

void arrange_by_type()
{
    for (int i = 0; i < (int)triangles.size(); i++)
    {
        triple &triangle = triangles[i];
        triple triangle_type =
            (triple){node_type[triangle.first], node_type[triangle.second], node_type[triangle.third]};

        triangles_by_type[triangle_type].push_back(triangle);
    }
}

void output()
{
    printf("%d\n", (int)triangles_by_type.size());
    for (std::map<triple, std::vector<triple>>::iterator it = triangles_by_type.begin(); it != triangles_by_type.end();
         it++)
    {
        std::vector<triple> triangles = it->second;
        printf("%d %d %d %d\n", node_type[triangles[0].first], node_type[triangles[0].second],
               node_type[triangles[0].third], (int)triangles.size());
        for (int i = 0; i < (int)triangles.size(); i++)
        {
            printf("%d %d %d\n", triangles[i].first, triangles[i].second, triangles[i].third);
        }
    }
}

void process()
{
    input();
    triangle_search();
    arrange_by_type();
    output();
}

int main(int argc, char *argv[])
{
    std::string dataset = std::string(argv[1]);

    std::string input_filename = "output/" + dataset + "_graph.txt";
    std::string output_filename = "output/" + dataset + "_triangles.txt";
    std::cout << input_filename << std::endl;

    freopen(input_filename.c_str(), "r", stdin);
    freopen(output_filename.c_str(), "w", stdout);

    process();

    return 0;
}