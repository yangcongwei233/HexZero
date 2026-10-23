// HexZero 批量搜索：使用 C 接口、双精度统计和单精度网络。
// 保持 Python/Numba 的 PUCT 运算顺序，禁用快速数学优化和乘加融合。
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
    int32_t terminal = 2; // 2 表示未检查，0 表示未结束，±1 表示规范视角的胜负。
    uint64_t tag = 0;
    bool expanded = false;
    double value = 0, edge_prior = 1;
    // 初始化节点的子节点索引为 −1，表示各动作尚未探索；其余数组使用成员默认零值。
    Node() { child.fill(-1); }
};
struct Tree {
    int n, actions;
    // 错误信息归当前树所有；避免在线程短暂存活的 Python 请求中
    // 注册动态库线程局部对象的析构函数。
    std::array<char, 1024> error{};
    std::vector<Node> nodes;
    std::vector<int32_t> leaves, evaluate;
    uint64_t batch = 0;
    // 建立指定边长的搜索树，校验原生实现支持的尺寸范围；节点存储按需增长。
    explicit Tree(int size) : n(size), actions(size * size) {
        if (size < 1 || size > 11) throw std::invalid_argument("native board size must be 1..11");
    }
    // 校验节点编号后返回可变引用，防止旧包装引用换根后失效的节点。
    Node& at(int id) {
        if (id < 0 || size_t(id) >= nodes.size()) throw std::out_of_range("stale native node");
        return nodes[id];
    }
    // 用栈上定长数组执行六邻接 BFS，判断指定棋色是否连接指定方向的两条边。
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
    // 结合真实/虚拟访问和先验计算 PUCT，返回最高分合法动作；根节点可使用带噪声的临时先验。
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
    // 按父节点动作创建后继节点，同时转置并换色为下一方视角；达到硬容量上限时拒绝扩展。
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
    // 批量选择路径，以临时虚拟访问分散同批搜索；无论成功或异常，都撤销本批预留计数。
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
    // 对已选叶子按批次标记去重，先执行终局检查，仅把非终局棋盘复制到调用方推理缓冲区。
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
    // 校验整批网络输出后写入叶节点先验和价值，再按原选择顺序逐路径回传；跨层翻转价值符号。
    void commit(const double* priors, const double* values, int count) {
        if (size_t(count) != evaluate.size()) throw std::invalid_argument("inference batch mismatch");
        // 修改树之前先校验整批输入。
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
        // 保留全部路径和回传顺序，包括到达重复叶子的路径。
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
    // 将实际动作的子树移到根位置，重新映射编号并释放无关分支；未探索落点则清空树。
    bool advance(int action) {
        if (action < 0 || action >= actions) throw std::invalid_argument("invalid move");
        int child = nodes.empty() ? -1 : nodes[0].child[action];
        leaves.clear(); evaluate.clear();
        if (child < 0) { nodes.clear(); return false; }
        // 仅保留可达子树，控制长对局的内存占用。
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
// 将错误文本截断写入当前树的固定缓冲区，保证末尾终止且不依赖线程局部析构。
void set_error(void* p, const char* message) noexcept {
    if (!p) return;
    auto& error = static_cast<Tree*>(p)->error;
    size_t length = std::min(std::strlen(message), error.size() - 1);
    std::memcpy(error.data(), message, length);
    error[length] = '\0';
}
// 包装原生调用并捕获异常，将错误写入树缓冲区后返回 −1，避免异常越过 C 接口。
template<class F> int guard(void* p, F f) noexcept {
    try { set_error(p, ""); return f(); }
    catch (const std::exception& e) { set_error(p, e.what()); return -1; }
    catch (...) { set_error(p, "unknown native search error"); return -1; }
}
// 校验不透明句柄非空并恢复原生树引用，供导出接口共用。
Tree& tree(void* p) { if (!p) throw std::invalid_argument("null native tree"); return *static_cast<Tree*>(p); }
}
// 返回接口版本，Python 加载时据此拒绝签名不兼容的旧动态库。
API int hx_abi() { return 2; }
// 读取树内最近错误文本；空句柄返回固定分配错误提示，调用方不负责释放字符串。
API const char* hx_error(void* p) {
    return p ? static_cast<Tree*>(p)->error.data() : "native tree allocation failed or invalid board size";
}
// 分配并返回不透明树句柄；构造失败返回空指针，不向调用方抛出 C++ 异常。
API void* hx_create(int n) noexcept {
    try { return new Tree(n); }
    catch (...) { return nullptr; }
}
// 释放树及全部节点存储；由 Python 终结回调调用，空指针可安全传入。
API void hx_destroy(void* p) noexcept { delete static_cast<Tree*>(p); }
// 清空搜索状态；非空棋盘指针用于建立校验后的根，空指针表示只清空。
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
// 返回当前节点数，供 Python 在选批前执行容量预算检查。
API int hx_size(void* p) { return guard(p, [&] { return int(tree(p).nodes.size()); }); }
// 将根作为单叶批次提交，供新树首次网络评估使用，不执行路径选择。
API int hx_root_batch(void* p) { return guard(p, [&] {
    auto& t = tree(p); t.at(0); t.leaves.assign(1, 0); return 0;
}); }
// 校验先验指针后选择指定数量的模拟路径，路径列表保留在原生树内。
API int hx_select(void* p, int count, const double* prior, double cpuct) { return guard(p, [&] {
    if (!prior) throw std::invalid_argument("null prior");
    tree(p).select(count, prior, cpuct); return 0;
}); }
// 检查输出缓冲区容量，复制去重后的待推理棋盘并返回实际网络批量大小。
API int hx_prepare(void* p, int8_t* boards, int capacity) { return guard(p, [&] {
    auto& t = tree(p);
    if (!boards || size_t(capacity) < t.leaves.size()) throw std::invalid_argument("small board buffer");
    return t.prepare(boards);
}); }
// 检查策略/价值指针，提交网络结果并完成全部路径回传；纯终局批次允许零个网络结果。
API int hx_commit(void* p, const double* priors, const double* values, int count) { return guard(p, [&] {
    if (count && (!priors || !values)) throw std::invalid_argument("null predictions");
    tree(p).commit(priors, values, count); return 0;
}); }
// 执行换根，返回 1 表示成功复用子树、0 表示已清空，异常返回负值。
API int hx_advance(void* p, int action) { return guard(p, [&] { return int(tree(p).advance(action)); }); }
// 仅返回副本，避免 NumPy 视图引用已释放或重新分配的原生存储。
// 将指定节点的棋盘、先验、边统计及状态复制到调用方数组；不会暴露原生存储视图。
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
