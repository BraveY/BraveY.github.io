git add .
$date = get-date -uformat "%Y-%m-%d %H:%M:%S"
git commit -m"new post $date"
git push origin
