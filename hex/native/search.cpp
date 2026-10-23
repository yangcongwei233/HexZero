// Batched HexZero search. C ABI, FP64 statistics, FP32 network unchanged.
// No fast-math/FMA contraction: preserve the Python/Numba PUCT operation order.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr int MAX_A = 121;
constexpr size_t MAX_NODES = 131072;
struct Node {
    std::array<int8_t, MAX_A> board{};
    std::array<double, MAX_A> prior{}, total{};
    std::array<int32_t, MAX_A> count{}, pending{}, child{}, order{};
    int32_t parent = -1, action = -1, visits = 0, pending_total = 0, used = 0;
    int32_t terminal = 2; // 2 = unknown, 0 = ongoing, +/-1 = canonical winner
    uint64_t tag = 0;
    bool expanded = false;
    double value = 0, edge_prior = 1;
    Node() { child.fill(-1); }
};
struct Tree {
    int n, actions;
    // Error text belongs to this tree. Do not register C++ thread-local
    // destructors in a DLL called by short-lived Python HTTP threads.
    std::array<char, 1024> error{};
    std::vector<Node> nodes;
    std::vector<int32_t> leaves, evaluate;
    uint64_t batch = 0;
    explicit Tree(int size) : n(size), actions(size * size) {
        if (size < 1 || size > 11) throw std::invalid_argument("native board size must be 1..11");
    }
    Node& at(int id) {
        if (id < 0 || size_t(id) >= nodes.size()) throw std::out_of_range("stale native node");
        return nodes[id];
    }
    bool connected(const Node& node, int stone, bool horizontal) const {
        std::array<uint8_t, MAX_A> seen{};
        std::array<int, MAX_A> queue{};
        int head = 0, tail = 0;
        for (int i = 0; i < n; ++i) {
            int a = horizontal ? i * n : i;
            if (node.board[a] == stone) { queue[tail++] = a; seen[a] = 1; }
        }
        constexpr int dr[] = {-1,-1,0,0,1,1}, dc[] = {0,1,-1,1,-1,0};
        while (head < tail) {
            int a = queue[head++], r = a / n, c = a % n;
            if ((horizontal ? c : r) == n - 1) return true;
            for (int k = 0; k < 6; ++k) {
                int y = r + dr[k], x = c + dc[k], b = y * n + x;
                if (y >= 0 && y < n && x >= 0 && x < n && !seen[b] && node.board[b] == stone) {
                    seen[b] = 1; queue[tail++] = b;
                }
            }
        }
        return false;
    }
    int choose(int id, const double* root_prior, double cpuct) {
        const Node& node = nodes[id];
        double scale = std::sqrt(std::max(1, node.visits + node.pending_total) + 1e-8);
        const double* prior = id == 0 ? root_prior : node.prior.data();
        double best = -std::numeric_limits<double>::infinity();
        int action = -1;
        for (int a = 0; a < actions; ++a) {
            if (node.board[a]) continue;
            int pending = node.pending[a], count = node.count[a] + pending;
            double q = count ? (node.total[a] - pending) / count : 0.;
            double score = q + cpuct * prior[a] * scale / (1 + count);
            if (score > best) { best = score; action = a; }
        }
        if (action < 0) throw std::runtime_error("no selectable edge");
        return action;
    }
    int create_child(int id, int action) {
        if (nodes.size() >= MAX_NODES) throw std::runtime_error("native search node limit reached");
        Node child;
        child.parent = id; child.action = action; child.edge_prior = nodes[id].prior[action];
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c) child.board[c*n+r] = -nodes[id].board[r*n+c];
        child.board[(action % n)*n + action/n] = -1;
        int index = int(nodes.size());
        nodes.push_back(std::move(child));
        nodes[id].child[action] = index;
        return index;
    }
    void select(int count, const double* prior, double cpuct) {
        if (nodes.empty() || !nodes[0].expanded || nodes[0].terminal != 0)
            throw std::runtime_error("root must be evaluated and nonterminal");
        if (count < 1 || count > 4096 || !std::isfinite(cpuct) || cpuct < 0)
            throw std::invalid_argument("invalid native batch or cpuct");
        leaves.clear(); evaluate.clear(); leaves.reserve(count);
        std::vector<std::pair<int,int>> reserved;
        reserved.reserve(size_t(count) * actions);
        try {
            for (int i = 0; i < count; ++i) {
                int id = 0;
                while (nodes[id].expanded && nodes[id].terminal == 0) {
                    int action = choose(id, prior, cpuct);
                    reserved.emplace_back(id, action);
                    nodes[id].pending[action]++; nodes[id].pending_total++;
                    int next = nodes[id].child[action];
                    id = next >= 0 ? next : create_child(id, action);
                }
                leaves.push_back(id);
            }
        } catch (...) {
            for (auto edge : reserved) { nodes[edge.first].pending[edge.second]--; nodes[edge.first].pending_total--; }
            leaves.clear(); throw;
        }
        for (auto edge : reserved) { nodes[edge.first].pending[edge.second]--; nodes[edge.first].pending_total--; }
    }
    int prepare(int8_t* boards) {
        ++batch; evaluate.clear();
        for (int id : leaves) {
            Node& node = nodes[id];
            if (node.tag == batch) continue;
            node.tag = batch;
            if (node.terminal == 2)
                node.terminal = connected(node, 1, false) ? 1 : (connected(node, -1, true) ? -1 : 0);
            if (node.terminal) { node.expanded = true; node.value = node.terminal; }
            else {
                std::memcpy(boards + evaluate.size()*actions, node.board.data(), actions);
                evaluate.push_back(id);
            }
        }
        return int(evaluate.size());
    }
    void commit(const double* priors, const double* values, int count) {
        if (size_t(count) != evaluate.size()) throw std::invalid_argument("inference batch mismatch");
        // Validate the entire batch before mutating the tree.
        for (int i = 0; i < count; ++i) {
            if (!std::isfinite(values[i])) throw std::invalid_argument("nonfinite network value");
            for (int a = 0; a < actions; ++a)
                if (!std::isfinite(priors[i*actions+a]) || priors[i*actions+a] < 0)
                    throw std::invalid_argument("invalid network prior");
        }
        for (int i = 0; i < count; ++i) {
            Node& node = nodes[evaluate[i]];
            std::copy_n(priors + i*actions, actions, node.prior.begin());
            node.value = values[i]; node.expanded = true;
        }
        // Keep all selected paths, including duplicate leaves; preserve backup order.
        for (int leaf : leaves) {
            double value = -nodes[leaf].value;
            int id = leaf;
            while (nodes[id].parent >= 0) {
                int parent = nodes[id].parent, action = nodes[id].action;
                Node& node = nodes[parent];
                if (!node.count[action]) node.order[node.used++] = action;
                node.count[action]++; node.total[action] += value; node.visits++;
                value = -value; id = parent;
            }
        }
        leaves.clear(); evaluate.clear();
    }
    bool advance(int action) {
        if (action < 0 || action >= actions) throw std::invalid_argument("invalid move");
        int child = nodes.empty() ? -1 : nodes[0].child[action];
        leaves.clear(); evaluate.clear();
        if (child < 0) { nodes.clear(); return false; }
        // Compact only the reachable subtree. This bounds memory over a long game.
        std::vector<int> order{child}, mapping(nodes.size(), -1);
        mapping[child] = 0;
        for (size_t i = 0; i < order.size(); ++i)
            for (int a = 0; a < actions; ++a) {
                int c = nodes[order[i]].child[a];
                if (c >= 0) { mapping[c] = int(order.size()); order.push_back(c); }
            }
        std::vector<Node> kept;
        kept.reserve(std::max(size_t(64), order.size()));
        for (int id : order) {
            kept.push_back(nodes[id]);
            Node& node = kept.back();
            node.parent = id == child ? -1 : mapping[node.parent];
            if (id == child) node.action = -1;
            for (int a = 0; a < actions; ++a) if (node.child[a] >= 0) node.child[a] = mapping[node.child[a]];
        }
        nodes.swap(kept);
        return true;
    }
};
void set_error(void* p, const char* message) noexcept {
    if (!p) return;
    auto& error = static_cast<Tree*>(p)->error;
    size_t length = std::min(std::strlen(message), error.size() - 1);
    std::memcpy(error.data(), message, length);
    error[length] = '\0';
}
template<class F> int guard(void* p, F f) noexcept {
    try { set_error(p, ""); return f(); }
    catch (const std::exception& e) { set_error(p, e.what()); return -1; }
    catch (...) { set_error(p, "unknown native search error"); return -1; }
}
Tree& tree(void* p) { if (!p) throw std::invalid_argument("null native tree"); return *static_cast<Tree*>(p); }
}
API int hx_abi() { return 2; }
API const char* hx_error(void* p) {
    return p ? static_cast<Tree*>(p)->error.data() : "native tree allocation failed or invalid board size";
}
API void* hx_create(int n) noexcept {
    try { return new Tree(n); }
    catch (...) { return nullptr; }
}
API void hx_destroy(void* p) noexcept { delete static_cast<Tree*>(p); }
API int hx_reset(void* p, const int8_t* board) { return guard(p, [&] {
    auto& t = tree(p); t.nodes.clear(); t.leaves.clear(); t.evaluate.clear();
    if (board) {
        Node root;
        for (int a = 0; a < t.actions; ++a) {
            if (board[a] < -1 || board[a] > 1) throw std::invalid_argument("invalid stone");
            root.board[a] = board[a];
        }
        t.nodes.push_back(root);
    }
    return 0;
}); }
API int hx_size(void* p) { return guard(p, [&] { return int(tree(p).nodes.size()); }); }
API int hx_root_batch(void* p) { return guard(p, [&] {
    auto& t = tree(p); t.at(0); t.leaves.assign(1, 0); return 0;
}); }
API int hx_select(void* p, int count, const double* prior, double cpuct) { return guard(p, [&] {
    if (!prior) throw std::invalid_argument("null prior");
    tree(p).select(count, prior, cpuct); return 0;
}); }
API int hx_prepare(void* p, int8_t* boards, int capacity) { return guard(p, [&] {
    auto& t = tree(p);
    if (!boards || size_t(capacity) < t.leaves.size()) throw std::invalid_argument("small board buffer");
    return t.prepare(boards);
}); }
API int hx_commit(void* p, const double* priors, const double* values, int count) { return guard(p, [&] {
    if (count && (!priors || !values)) throw std::invalid_argument("null predictions");
    tree(p).commit(priors, values, count); return 0;
}); }
API int hx_advance(void* p, int action) { return guard(p, [&] { return int(tree(p).advance(action)); }); }
// Copies only: no NumPy view can outlive/refer into reallocated native storage.
API int hx_snapshot(void* p, int id, int8_t* board, double* prior, double* totals,
                    int32_t* counts, int32_t* children, int32_t* order, int32_t* info, double* edge_prior) {
    return guard(p, [&] {
        auto& t = tree(p); auto& v = t.at(id);
        std::copy_n(v.board.data(), t.actions, board);
        std::copy_n(v.prior.data(), t.actions, prior);
        std::copy_n(v.total.data(), t.actions, totals);
        std::copy_n(v.count.data(), t.actions, counts);
        std::copy_n(v.child.data(), t.actions, children);
        std::copy_n(v.order.data(), t.actions, order);
        info[0] = v.parent; info[1] = v.action; info[2] = v.visits; info[3] = v.terminal;
        info[4] = v.expanded; info[5] = v.used; info[6] = v.pending_total;
        *edge_prior = v.edge_prior;
        return 0;
    });
}
