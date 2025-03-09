import sys
tag = sys.argv[1]
head = """
<!DOCTYPE html>
<html>
  <head>
    <title>Redirecting to main branch</title>
    <meta charset="utf-8" />
"""

line1 = r'    <meta http-equiv="refresh" content="0; url=./' + f"{tag}" + r'/index.html" />' + "\n" 
line2 = r'    <link rel="canonical" href="https://cytnx-dev.github.io/Cytnx_doc/' + f"{tag}" + r'/index.html" />' + "\n"

end = """
  </head>
</html>
"""
print(head + line1 + line2 + end)


