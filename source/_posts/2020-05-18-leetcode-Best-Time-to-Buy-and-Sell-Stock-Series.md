---
title: leetcode Best Time to Buy and Sell Stock Series
date: 2020-05-18 15:42:02
categories: 题解
tags:
- 动态规划
copyright: true
---

## 题意

### 只能买一次

[题目链接](<https://leetcode.com/problems/best-time-to-buy-and-sell-stock/> )

Say you have an array for which the *i*th element is the price of a given stock on day *i*.

If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Note that you cannot sell a stock before you buy one.

**Example 1:**

```
Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
```

**Example 2:**

```
Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
```

#### 思路

在遍历的时候记录一个当前为止的最低价格`min_price`,然后用当前价格去减去最低价格得到当前价格能够获得的最大利润，最后返回每天的最大利润中的最大。

#### 复杂度

遍历一次$O(n)$

#### 代码

```cc
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        const int len = prices.size();
        if(len<1) return 0;
        vector<int> min_price(len);
        vector<int> max_profit(len);
        min_price[0]=prices[0];
        max_profit[0]=0;
        for(int i=1; i<len; i++){
            min_price[i]=min(min_price[i-1],prices[i]);
            max_profit[i]=max(max_profit[i-1],prices[i]-min_price[i]);
        }
        return max_profit[len-1];
    }
};
```

### 不限次数

[题目链接](<https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/> )

#### 思路

直接采用贪心的思路，如果价格比前一天的价格高就卖掉，否则就买入，贪心地去买低价格以及卖高价格。

#### 复杂度

遍历一次$O(n)$

#### 代码

```cc
class Solution {
public:
    int maxProfit(vector<int>& prices) {
    	int len = prices.size();
    	if(len<=0) return 0 ;
        int buy = prices[0];
        int profit = 0;
        for(int i=0; i<len; i++){
        	if(buy < prices[i]){
        		profit += prices[i] - buy;
        	}
        	buy = prices[i];
        }
        return profit;
    }
private:
};
```

### 只能买两次

[题目链接](<https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/> )

#### 思路

可以看成是划分成两段后，分别对左边这一段用只买一次的思路得到左边的最大利润`left_profit`，然后右边这一段也是用题目一的思路只买一次得到右边的最大利润`right_profit`。之后两段相加就得到了从当前位置划分所得到的最终最大利润，这样遍历的时候就得到每个位置的最大利润了，最后对所有位置取最大的就可以了。

上述的思路会造成很多重复计算，可以继续优化下，直接遍历两次分别记录左边与右边可以获得的最大利润。左边部分肯定先执行买的操作，所以从左到右遍历，记录一个最小价格，然后在当前遍历节点考虑卖出能否获得最大收益，可以获得就更新最大收益，否则保持原先的最大收益。右边部分，肯定最后执行卖的操作，所以从右到左记录一个最大的价格，对每个遍历节点记录当前买入能否获得最大利润，可以就更新最大收益，不可以就保持之前后面的最大收益。

最后再次遍历来获得每个节点的最终最大收益。

三次遍历，用状态转移方程来表达：
$$
Profit_i^{left} = max(Profit_{i-1}^{left}, Price[i]-minPrice) \\
Profit_i^{right} = max(Profit_{i+1}^{right}, Price[i]-maxPrice) \\
Profit_i^{final} = max(Profit_{i-1}^{final}, Profit_i^{left}+Profit_i^{right})
$$


#### 复杂度

3次for循环遍历：$O(n)$

#### 代码

```cc
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        const int len = prices.size();
        if(len<1) return 0;
        int min_price=prices[0];
        int max_price=prices[len-1];
        vector<int> left_profit(len);
        vector<int> right_profit(len);
        left_profit[0]=0;
        right_profit[len-1]=0;
        for(int i=1;i<len;i++){
        	min_price = min(min_price, prices[i]);
        	left_profit[i] = max(left_profit[i-1],prices[i]-min_price);
        }
        for(int i=len-2;i>=0;i--){
        	max_price = max(max_price,prices[i]);
        	right_profit[i] = max(right_profit[i+1], max_price-prices[i]);
        }
        int final_max_profit=0;
        for(int i=1;i<len;i++){
        	final_max_profit = max(final_max_profit,left_profit[i]+right_profit[i]);
        }
        return final_max_profit;
    }
};
```



### 只能买K次

#### 思路

首先需要知道，给定n天，最多能执行买卖的次数只要n/2次，所以当k大于n/2的时候就相当于可以执行不限次数的买卖，也就是前面的贪心方法了。

采用动态规划的思路，使用`dp[i][k]`表示在前i天最多买卖k次所获得的最大收益。

在第i天如果不执行操作的话，最大收益同前一天一样即：`dp[i][k] = dp[i-1][k]`，如果执行操作的话只能在第i天卖出，同时前面的交易次数需要减少一次,假设在第j天买入，这样收益就变成了两部分第j天以前的收益`dp[j][k-1]`与在第j天买入，第i天卖出的收益`price[i]-price[j]`。所以卖出的总收益就是`dp[j][k-1]+price[i]-price[j]` ,所以状态转移方程：
$$
dp[i][k]=max(dp[i-1][k],max(dp[j][k-1]+price[i]-price[j])),j<i
$$
得到这个式子后，为了确定卖出所能够获得的最大收益，最暴力的方法就是对第0至第i-1天，再进行遍历来得到，卖出的最大收益。

这就是我之前的代码：

```cc
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        if(prices.empty()) return 0;
        int ans = 0;
        if(k >= prices.size() / 2){
            for(int i = 1; i < prices.size(); ++i){
                if(prices[i] > prices[i - 1])
                    ans += prices[i] - prices[i - 1];
            }
            return ans;
        }
        vector<vector<int>> dp(prices.size(), vector<int>(k + 1, 0));
        for(int x = 1; x <= k; ++x){
            for(int i = 1; i < prices.size(); ++i){
                dp[i][x] = dp[i - 1][x]; // 不做交易
                for(int j = 0; j < i; ++j){
                    dp[i][x] = max(dp[i][x], dp[j][x - 1] + prices[i] - prices[j]);
                }
            }
            ans = max(ans, dp[prices.size() - 1][x]);
        }
        return ans;
    }
};
```

时间复杂度就是$O(n^2k)$

这样会存在很多次重复计算，比如第5天计算前4天卖出的最大收益，在计算第6,天等又会重复计算前4天卖出的收益。

进一步优化，将前面计算的结果存储起来。具体思路是：选择第j天卖出的收益可以转换成`price[i]-(price[j]-dp[j][k-1])`这样设置一个`min_diff= price[j]-dp[j][k-1]`变量，在遍历i的时候就可以更新这个遍历得到前i-1天所能够得到最小值了。具体看代码

#### 复杂度

优化后只用两层循环，所以时间复杂度是$O(nk)$,空间复杂度是$O(nk)$

#### 代码

```cc

class Solution {
  public:
	int maxProfit(int k, vector<int>& prices) {
		if (prices.empty()) return 0;
		int ans = 0;
		if (k >= prices.size() / 2) {
			for (int i = 1; i < prices.size(); ++i) {
				if (prices[i] > prices[i - 1])
					ans += prices[i] - prices[i - 1];
			}
			return ans;
		}
		vector<vector<int>> dp(prices.size(), vector<int>(k + 1, 0));
		for (int x = 1; x <= k; ++x) {
			int min_diff = prices[0]; //第0天的情况就是price[0] - 0 = prices[0]
			for (int i = 1; i < prices.size(); ++i) { //从第1天开始，
				dp[i][x] = max(dp[i - 1][x], prices[i] - min_diff);// 卖出所得到的收益
				min_diff = min(min_diff, prices[i] - dp[i][x - 1]);//加入当前天数后得到最新的最小差值。
			}
		}
		return dp[prices.size() - 1][k];
	}
};
```

## 参考

<https://leetcode.wang/leetcode-123-Best-Time-to-Buy-and-Sell-StockIII.html> 