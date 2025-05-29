uT_new = uT.relabel(old_label="a", new_label="xx")
uT.print_diagram()
uT_new.print_diagram()

print(uT_new.same_data(uT))
