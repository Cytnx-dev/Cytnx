uT_new = uT.relabel("a","xx")
uT.print_diagram()
uT_new.print_diagram()

print(uT_new.same_data(uT))
