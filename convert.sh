

echo "Converting Markdown → Confluence-ready HTML..."

find . -name "*.md" | while read file; do
    out="${file%.md}.html"

    pandoc "$file" \
        -o "$out" \
        --standalone \
        --mathjax \
        --highlight-style=tango

    echo "Converted: $file → $out"
done

echo "Done."
