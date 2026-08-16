#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <unordered_map>

struct FastTrieNode {
    int32_t token_id;
    std::unordered_map<uint8_t, int32_t> children;
    FastTrieNode() : token_id(-1) {}
};

static std::vector<FastTrieNode> trie_nodes;

static PyObject* init_trie(PyObject* self, PyObject* args) {
    PyObject* token_list;
    if (!PyArg_ParseTuple(args, "O", &token_list)) return NULL;
    
    trie_nodes.clear();
    trie_nodes.reserve(4000000);
    trie_nodes.emplace_back(); // root node 0
    
    // Base 256 bytes
    for (int b = 0; b < 256; ++b) {
        int next_idx = trie_nodes.size();
        trie_nodes.emplace_back();
        trie_nodes[next_idx].token_id = b;
        trie_nodes[0].children[(uint8_t)b] = next_idx;
    }
    
    Py_ssize_t n = PyList_Size(token_list);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PyList_GetItem(token_list, i);
        long tid = PyLong_AsLong(PyTuple_GetItem(item, 0));
        PyObject* bobj = PyTuple_GetItem(item, 1);
        char* buf;
        Py_ssize_t len;
        PyBytes_AsStringAndSize(bobj, &buf, &len);
        
        int curr = 0;
        for (Py_ssize_t j = 0; j < len; ++j) {
            uint8_t b = (uint8_t)buf[j];
            auto it = trie_nodes[curr].children.find(b);
            if (it == trie_nodes[curr].children.end()) {
                int next_idx = trie_nodes.size();
                trie_nodes.emplace_back();
                trie_nodes[curr].children[b] = next_idx;
                curr = next_idx;
            } else {
                curr = it->second;
            }
        }
        trie_nodes[curr].token_id = (int32_t)tid;
    }
    
    return PyLong_FromSize_t(trie_nodes.size());
}

static PyObject* encode_fast(PyObject* self, PyObject* args) {
    const char* text;
    Py_ssize_t text_len;
    if (!PyArg_ParseTuple(args, "s#", &text, &text_len)) return NULL;
    
    std::vector<uint32_t> tokens;
    tokens.reserve(text_len / 2 + 16);
    
    Py_ssize_t idx = 0;
    while (idx < text_len) {
        int curr = 0;
        int32_t longest_id = -1;
        Py_ssize_t longest_len = 0;
        
        Py_ssize_t scan = idx;
        while (scan < text_len) {
            uint8_t b = (uint8_t)text[scan];
            auto it = trie_nodes[curr].children.find(b);
            if (it == trie_nodes[curr].children.end()) break;
            curr = it->second;
            scan++;
            if (trie_nodes[curr].token_id >= 0) {
                longest_id = trie_nodes[curr].token_id;
                longest_len = scan - idx;
            }
        }
        
        if (longest_id >= 0 && longest_len > 0) {
            tokens.push_back((uint32_t)longest_id);
            idx += longest_len;
        } else {
            tokens.push_back((uint32_t)(uint8_t)text[idx]);
            idx += 1;
        }
    }
    
    return PyBytes_FromStringAndSize((const char*)tokens.data(), tokens.size() * sizeof(uint32_t));
}

static PyMethodDef FastBPEMethods[] = {
    {"init_trie", init_trie, METH_VARARGS, "Initialize fast C++ Trie"},
    {"encode_fast", encode_fast, METH_VARARGS, "Encode text to uint32 token buffer at C++ speed"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fast_bpe_module = {
    PyModuleDef_HEAD_INIT,
    "aglm_fast_bpe",
    "High-speed C++ Trie Tokenizer Accelerator",
    -1,
    FastBPEMethods
};

PyMODINIT_FUNC PyInit_aglm_fast_bpe(void) {
    return PyModule_Create(&fast_bpe_module);
}
