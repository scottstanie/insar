# coding: utf-8
def redo_links(directory, search, old_replace, new_replace):
    for g in glob.glob(os.path.join(dd, '*' + search)):
        fg = os.path.abspath(g)
        if not os.path.exists(fg):
            old_link = os.readlink(fg)
            new_link = old_link.replace(old_replace, new_replace)
            os.unlink(fg)
            os.symlink(new_link, fg)
    
