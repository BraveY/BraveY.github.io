---
title: leetcode 134 Gas Station
date: 2020-04-24 13:55:23
categories: 题解
tags:
- 贪心
copyright: true
---

## 题意

[题目链接](<https://leetcode.com/problems/gas-station/> )  给出一个加油站容量数组，和路途消耗数组，问是否可以完成一次循环。

There are *N* gas stations along a circular route, where the amount of gas at station *i* is `gas[i]`.

You have a car with an unlimited gas tank and it costs `cost[i]` of gas to travel from station *i* to its next station (*i*+1). You begin the journey with an empty tank at one of the gas stations.

Return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1.

**Note:**

- If there exists a solution, it is guaranteed to be unique.
- Both input arrays are non-empty and have the same length.
- Each element in the input arrays is a non-negative integer.

**Example 1:**

```
Input: 
gas  = [1,2,3,4,5]
cost = [3,4,5,1,2]

Output: 3

Explanation:
Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 4. Your tank = 4 - 1 + 5 = 8
Travel to station 0. Your tank = 8 - 2 + 1 = 7
Travel to station 1. Your tank = 7 - 3 + 2 = 6
Travel to station 2. Your tank = 6 - 4 + 3 = 5
Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
Therefore, return 3 as the starting index.
```

**Example 2:**

```
Input: 
gas  = [2,3,4]
cost = [3,4,3]

Output: -1

Explanation:
You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
Let's start at station 2 and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 0. Your tank = 4 - 3 + 2 = 3
Travel to station 1. Your tank = 3 - 3 + 3 = 3
You cannot travel back to station 2, as it requires 4 unit of gas but you only have 3.
Therefore, you can't travel around the circuit once no matter where you start.
```

## 思路

贪心的方式就是每到一个加油站就加满所有的油。

### 暴力枚举

按照题意叙述，可以对每个起点进行一次判定，判定方法就是题目叙述样例的方式，每次都加满油然后走下一段路程，如果油量不够下一段路程则返回-1，如果可以完成路程的话继续走下一段。

**复杂度**

需要两层for循环因此是$O(n^2)$ 

### 遍历

仔细思考题目，可以得到一个结论，只有当gas数组之和大于cost数组的时候才可能有解，也就是只有加油站的总油量大于路程所需的消耗量的时候才能完成路程。

将gas数组和cost数组相减可以得到每个站点能否完成下一段的值，如果gas[i]-cost[i]的值非负，说明在这个站点中加的油足够完成下一段路程。如果值为负，说明不能完成下一段路程，但是在到达这个站点的时候可能有之前站点剩余的油量。

因此从第0个站点开始，到最后一个站点截止。将每个站点的gas[i]-cost[i]的值进行累加，如果累加值大于等于0 ，说明从出发点到当前点的油量足够。如果累加上当前站点的gas[i]-cost[i]的值变为负数了，说明当前站点不能到达下一个站点。当累积值为负的时候，将下个站点作为起点，然后累加值置为0 重新开始累加值的计算。

![](https://res.cloudinary.com/bravey/image/upload/v1587711117/blog/coding/lc134.jpg )

这样一来就会将站点分为k段，前k-1段的累加值都是负数，如果全程的累加值sum(0,n-1)大于0，则说明有解。因为：
$$
sum(0,n-1) = cur(0,i-1) +cur(i,j-1)+\dots+cur(k,n-1)
$$
所以当总和非负的时候，因为前面k-1段都是负值，所以最后一段一定为正数。也就是最后段的剩余油量足够填补前面k-1段的不足油量，从而完成整个路程。

**复杂度**

因为只需一层循环所以时间复杂度是$O(n)$

## 代码

### 暴力枚举

```cc
/*
Runtime: 208 ms, faster than 10.35% of C++ online submissions for Gas Station.
Memory Usage: 7 MB, less than 100.00% of C++ online submissions for Gas Station.
 */
class Solution {
 public:
	int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
		int n = gas.size();
		for (int i = 0; i < n; i++) {
			int gas_left = 0;
			int index = i;
			int count = 0;
			while (gas[index] + gas_left >= cost[index]) {
				if (count == n) break;
				gas_left = gas_left + gas[index] - cost[index];
				if (index + 1 < n )
					index++;
				else index = 0;
				count++;
			}
			if (count == n) return i;
		}
		return -1;
	}

 private:
};
```

### 遍历

```cc
/*
Runtime: 4 ms, faster than 98.11% of C++ online submissions for Gas Station.
Memory Usage: 7 MB, less than 100.00% of C++ online submissions for Gas Station.
 */
class Solution2 {
 public:
	int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
		int n = gas.size();
		int cur_gas = 0;
		int sum = 0;
		int start = 0;
		for (int i = 0; i < n; i++) {
			cur_gas += gas[i] - cost[i];
			sum += gas[i] - cost[i];
			if (cur_gas < 0) {
				cur_gas = 0;
				start = i + 1;
			}
		}
		if (sum < 0) return -1;
		else return start;
	}

 private:
};
```

## 参考

<https://blog.csdn.net/Irving_zhang/article/details/78367216> 