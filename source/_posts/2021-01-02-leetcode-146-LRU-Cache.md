---
title: leetcode 146 LRU Cache
date: 2021-01-02 20:27:10
categories: 题解
tags:
- Hash
- 链表
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/lru-cache/)

Design a data structure that follows the constraints of a **[Least Recently Used (LRU) cache](https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU)**.

Implement the `LRUCache` class:

- `LRUCache(int capacity)` Initialize the LRU cache with **positive** size `capacity`.
- `int get(int key)` Return the value of the `key` if the key exists, otherwise return `-1`.
- `void put(int key, int value)` Update the value of the `key` if the `key` exists. Otherwise, add the `key-value` pair to the cache. If the number of keys exceeds the `capacity` from this operation, **evict** the least recently used key.

**Follow up:**
Could you do `get` and `put` in `O(1)` time complexity?

 

**Example 1:**

```
Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4
```

 

**Constraints:**

- `1 <= capacity <= 3000`
- `0 <= key <= 3000`
- `0 <= value <= 104`
- At most `3 * 104` calls will be made to `get` and `put`.

## 哈希+链表

### 思路

使用链表来维护key的使用顺序，每次新使用的key插入到链表的最前方，每次需要evcit的时候，删除链表最后面的key。

### 复杂度

时间复杂度$O(1)$

空间复杂度$O(N)$

### 代码

```cc
/*
Runtime: 140 ms, faster than 73.18% of C++ online submissions for LRU Cache.
Memory Usage: 43.3 MB, less than 12.74% of C++ online submissions for LRU Cache.
使用链表的有序来进行排序，而不重新使用排序来得到。
*/
class LRUCache {
  public:
    LRUCache(int capacity) {
        _capacity = capacity;
    }
    
    int get(int key) {
        if(!kv.count(key)) return -1;
        updateLRU(key);
        return kv[key];
    }
    
    void put(int key, int value) {
        if(!kv.count(key) && kv.size() == _capacity) {
            evict();
        }
        updateLRU(key);//updateLRU()的逻辑会先判断kv里面是否存在key,所以必须先执行updateLRU的操作，再往kv中插入，顺序不能乱。
        kv[key] = value;        
    }

  private:
    int _capacity;
    list<int> lru; //链表按序存放使用的key,新使用的放在最前面
    unordered_map<int, list<int>::iterator> mp; //记录key在链表的位置。
    unordered_map<int, int> kv;

    void updateLRU(int key) { 
        if (kv.count(key)){ //在kv中的key必须保证存在于lru链表中，所以必须先更新lru再插入新的key在kv中。
            lru.erase(mp[key]); //已经存在的key需要删除后提到最前。
        }
        lru.push_front(key); 
        mp[key] = lru.begin();
    }
    void evict(){
        mp.erase(lru.back());
        kv.erase(lru.back());
        lru.pop_back();
    }
};
```

## 总结

链表自身就是有序的，所以可以维护这个数据结构来完成需要排序的操作。而不是进行排序算法。

## 参考

[lc](https://leetcode.com/problems/lru-cache/discuss/45976/C%2B%2B11-code-74ms-Hash-table-%2B-List)