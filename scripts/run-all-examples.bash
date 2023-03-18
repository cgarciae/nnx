set -e

for f in $(find examples -name "*.py"); do
    echo -e "\n---------------------------------"
    echo "$f"
    echo "---------------------------------"
    poetry run python "$f"
done
