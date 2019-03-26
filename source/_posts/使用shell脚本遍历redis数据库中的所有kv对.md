## 使用shell脚本遍历redis数据库中的所有kv对

````
#!/bin/bash
filename='redis'`date +%Y-%m-%d_%H:%M`
work_path=$(dirname "$0") 
echo "实例化redis数据文件为:$work_path/$filename"
echo "keys *" | redis-cli > key_db.txt
echo "将所有key保存到:$work_path/key_db.txt"
for line in `cat key_db.txt`
do
        echo "key:$line " >>$work_path/$filename.txt
        echo "key-value:" >>$work_path/$filename.txt
        echo "hgetall $line" | redis-cli >>$work_path/$filename.txt
done
````

