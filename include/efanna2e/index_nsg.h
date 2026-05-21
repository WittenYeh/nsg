#ifndef EFANNA2E_INDEX_NSG_H
#define EFANNA2E_INDEX_NSG_H

#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include <cassert>
#include <random>
#include <unordered_map>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <stack>

namespace efanna2e {

class IndexNSG : public Index {
 public:
  explicit IndexNSG(const size_t dimension, const size_t n, Metric m, Index *initializer);


  virtual ~IndexNSG();

  virtual void Save(const char *filename)override;
  virtual void Load(const char *filename)override;


  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override;
  void SearchWithOptGraph(
      const float *query,
      size_t K,
      const Parameters &parameters,
      unsigned *indices);
  void OptimizeGraph(float* data);

  // WittenYeh fork: read-only accessors used by the nsg-comparison profiler
  // to dump post-build graph stats (degree dist, edge sets, ep_).
  unsigned probe_ep() const { return ep_; }
  const std::vector<std::vector<unsigned>>& probe_final_graph() const {
    return final_graph_;
  }

  // WittenYeh fork: write-side hooks used by the cross-search experiment to
  // inject an externally-built NSG (e.g., one produced by faiss::IndexNSG)
  // and then drive efanna's SearchWithOptGraph against it. Caller must:
  //   1. construct IndexNSG with the correct (dimension, n, metric, nullptr)
  //   2. set_data(xb), set_final_graph(...), set_ep(...)
  //   3. OptimizeGraph(xb)  -> SearchWithOptGraph(...)
  void inject_data(const float* data) { data_ = data; has_built = true; }
  void inject_ep(unsigned ep) { ep_ = ep; }
  void inject_final_graph(std::vector<std::vector<unsigned>>&& g) {
    final_graph_ = std::move(g);
  }

  protected:
    typedef std::vector<std::vector<unsigned > > CompactGraph;
    typedef std::vector<SimpleNeighbors > LockGraph;
    typedef std::vector<nhood> KNNGraph;

    CompactGraph final_graph_;

    Index *initializer_ = nullptr;
    void init_graph(const Parameters &parameters);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        boost::dynamic_bitset<>& flags,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    //void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph& cut_graph_);
    void InterInsert(unsigned n, unsigned range, std::vector<std::mutex>& locks, SimpleNeighbor* cut_graph_);
    void sync_prune(unsigned q, std::vector<Neighbor>& pool, const Parameters &parameter, boost::dynamic_bitset<>& flags, SimpleNeighbor* cut_graph_);
    void Link(const Parameters &parameters, SimpleNeighbor* cut_graph_);
    void Load_nn_graph(const char *filename);
    void tree_grow(const Parameters &parameter);
    void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter);


  private:
    unsigned width;
    unsigned ep_;
    std::vector<std::mutex> locks;
    char* opt_graph_ = nullptr;
    size_t node_size;
    size_t data_len;
    size_t neighbor_len;
    KNNGraph nnd_graph;
    // WittenYeh fork: class-level RNG with the same seed FAISS uses for its
    // NSG::rng member (0x0903). Drives ep_ pick + findroot random fallback.
    // mt19937 is byte-identical to faiss::RandomGenerator (both wrap mt19937
    // and use `mt() % n` for rand_int(n)).
    std::mt19937 mt_rng_{0x0903};
};
}

#endif //EFANNA2E_INDEX_NSG_H
