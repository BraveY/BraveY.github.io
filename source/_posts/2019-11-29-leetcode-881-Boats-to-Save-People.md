---
title: leetcode 881 Boats to Save People
date: 2019-11-29 15:49:08
categories: 题解
tags:
- 贪心
- leetcode
copyright: true
---

# leetcode 881 Boats to Save People

[题目来源](<https://leetcode.com/problems/boats-to-save-people/> )。要求在船有载重量和人数限制为2的情况下，给出将所有人运过河的最小船数。

<!--more-->

## 思路

使用贪心算法，进行求解。

### 桶排序 

首先是自己的思路：贪心的考虑每只船都尽可能的装满限重，然后优先让胖的人先上船，之后寻找在剩下载重量限制的情况下，找最胖的人。

在具体实现上，先遍历一遍人数的重量，按照重量进行桶排序，因为重量不会超过limit，而且都是整数所以是可以实现的。每个桶记录该重量下的人数，然后先将重量最大的给安排上船，并寻找能够匹配的第二个重量。直到最轻的被遍历完得到答案。

## 双指针

提交后发现网上的思路和自己有一点差别：配对的贪心规则是最胖的先走之后，选择最轻的进行匹配。这样就很容易的使用双指针来实现。不过为什么两种规则都能够过，自己还在思考中,参考链接中有对这种贪心规则的证明。

## 代码

### 桶排序

```cpp
class Solution {
public:
	/* 桶排序 贪心的选择最胖的先走
	Runtime: 96 ms
	Memory Usage: 16.5 MB*/
    int numRescueBoats(vector<int>& people, int limit) {
        int num = people.size();
		if(num<=0) return 0;
        vector<int> person(limit+1, 0);
        for(int i=0; i<num; i++){
            person[people[i]]++;
        }
		int ans = 0;
		// int crossed = 0;
		for(int i = limit; i>0; i--){
			for(int j=person[i]; j>0; j--){
				ans++;
				// crossed++;
				person[i]--;// 运走一个人后 计数减1
				int rest = limit - i;
				for(int k=rest; k>0; k--){
					if(person[k]>0){
						person[k]--;
						if(k==i) j--; // 如果减去的和当前指向的weight一样 则需要减去自身
						// crossed++;
						break;
					}
				}
			}
		}
		// if(crossed==num) cout<<"all crossed" <<endl;
		return ans;   
    }
};
```

需要注意的是如果剩下的重量和当前重量一致的时候，迭代的计数也要减1，比如limit为4，当前迭代的重量为2，剩下的匹配重量也为2情况。

### 双指针

```cpp
class Solution {
public:
	/*排序后 双指针
	Runtime: 116 ms
	Memory Usage: 13.6 MB
    */
    int numRescueBoats(vector<int>& people, int limit){
    	sort(people.begin(), people.end());
    	int ans = 0;
    	int i = 0;
    	int j = people.size() - 1;
    	while(i<=j){
    		if(people[i]+people[j]<=limit){
    			i++;
    			j--;
    			ans++;
    		}
    		else{
    			j--;
    			ans++;
    		}
    	}
    	return ans;
    }
};
```

## 复杂度

leetcode上面显示桶排序的速度能够超过96%的提交，但是空间使用仅超过11%.

### 桶排序

桶排序的时间复杂读为O(n)，但是空间消耗较多为O(limit)。自己实现的代码中虽然有三层for循环，但是每一层都不是人数n，从总的被遍历到的人来考虑，前面两层是遍历的人数因此是O(n)，而每一个重量的人都需要去寻找匹配的重量，考虑最坏的情况是O(limit)，但是一般来说不会寻找太多次，所以可以认为近似的是O(1)。因此总得时间复杂度为O(n)，空间复杂度为O(limit)

### 双指针

双指针的时间复杂度为排序的时间复杂度，使用快排的话是O(nlogn)。不需要额外的空间，所以是O(1)。

## 参考

对于第二种匹配规则的贪心的证明[参考](<https://zhanghuimeng.github.io/post/leetcode-881-boats-to-save-people/>  ) 但是存疑，因为我用第一种的匹配规则也做出来了。