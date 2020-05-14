---
title: leetcode 207 Course Schedule
date: 2020-05-13 14:41:02
categories: 题解
tags:
- BFS
- DFS
copyright: true
---

## 题意

[题目链接](<https://leetcode.com/problems/course-schedule/> ) 实际就是有向图判断有没有环的问题，也就是拓扑排序。

There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses-1`.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: `[0,1]`

Given the total number of courses and a list of prerequisite **pairs**, is it possible for you to finish all courses?

 

**Example 1:**

```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0. So it is possible.
```

**Example 2:**

```
Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0, and to take course 0 you should
             also have finished course 1. So it is impossible.
```

 

**Constraints:**

- The input prerequisites is a graph represented by **a list of edges**, not adjacency matrices. Read more about [how a graph is represented](https://www.khanacademy.org/computing/computer-science/algorithms/graph-representation/a/representing-graphs).
- You may assume that there are no duplicate edges in the input prerequisites.
- `1 <= numCourses <= 10^5`

自己最开始想的办法是利用BFS加上访问标记来判断是否有环，这种方法判断的是无向图是否 有环的方法，所以不能通过全部样例。

## DFS

### 思路

自己最开始以为这种非经典树结构不能使用DFS就没有往下面想了。整体的思路就是在DFS过程中设计一个数组来存储访问标志，visited和visiting，如果在DFS过程中访问到一个visiting的节点，说明图中有环需要返回false。

需要注意的是，可能形成多个独立的子图，因此不是一次性DFS遍历后就可以更新了所有节点的状态，所以需要对每个节点都DFS遍历一遍，如果一个子图中出现了环就返回false。

### 代码

```cc
/*
Runtime: 40 ms, faster than 45.34% of C++ online submissions for Course Schedule.
Memory Usage: 13.2 MB, less than 90.91% of C++ online submissions for Course Schedule.
 */
class Solution2 {
 public:
	bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
		int n = prerequisites.size();
		graph_ = vector<vector<int>> (numCourses); //init
		for (int i = 0; i < n; i++) {
			graph_[prerequisites[i][0]].push_back(prerequisites[i][1]);
		}
		// 0 :unk, 1 :visiting, 2 :visited
		vector<int> v(numCourses, 0);
		for (int i = 0; i < numCourses; i++) {
			if (dfs(i, v)) return false;
		}
		return true;
	}

 private:
	vector<vector<int>> graph_;
	bool dfs(int cur, vector<int>& v) {
		if (v[cur] == 1) return true; //cycle
		if (v[cur] == 2) return false; // no cycle
		v[cur] = 1;
		for (const int t : graph_[cur])//如果没有相邻的边直接跳过循环
			if (dfs(t, v)) return true;
		v[cur] = 2;
		return false;
	}
};
```

### 复杂度

时间复杂度：O(N) N为节点数

## BFS

### 思路

BFS遍历的时候是从入度为0的节点开始，将其邻居节点的入读减1，然后通过不断删除入度为0的节点来进行遍历的。因为当出现环的时候环中的所有节点入度不小于1，因此不会被遍历到。因此整个遍历完成后，待遍历的节点如果还大于0，说明有环，返回false。

这种思路能够得到形成环的节点数。

### 代码

```cc
/*
BFS
Runtime: 40 ms, faster than 45.34% of C++ online submissions for Course Schedule.
Memory Usage: 13.2 MB, less than 90.91% of C++ online submissions for Course Schedule.
 */
class Solution {
 public:
	bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
		vector<vector<int>> graph_(numCourses);
		int n = prerequisites.size();
		vector<int> degree(numCourses, 0);
		for (int i = 0; i < n; i++) {
			graph_[prerequisites[i][0]].push_back(prerequisites[i][1]);
			degree[prerequisites[i][1]]++; // in degree 入度表示需要先修的课的门数
		}
		queue<int> q;
		for (int i = 0; i < numCourses; i++)
			if (degree[i] == 0) q.push(i); //先从不需要先修课的开始遍历
		while (!q.empty()) {
			int curr = q.front(); q.pop(); numCourses--;//将待遍历的节点减1
			for (auto next : graph_[curr])
				if (--degree[next] == 0) q.push(next);//邻居节点的入度减1
		}
		return numCourses == 0; // 有环的节点入度始终大于0，不会经过while循环，使得待遍历的数大于1.
	}
};
```

### 复杂度

时间复杂度也是O(N) N为节点数。

## 参考

<https://www.bilibili.com/video/av38557948> 

<https://leetcode.wang/leetcode-207-Course-Schedule.html> 

<https://leetcode.com/problems/course-schedule/discuss/58509/C%2B%2B-BFSDFS> 