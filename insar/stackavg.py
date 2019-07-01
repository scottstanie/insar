# TODO: make simple stack averaging work
# from shutil import copyfile
# import matplotlib.pyplot as plt
# import glob

# def avg_stack(igram_path, row, col):
#     def plot_deformation(img, title='', cbar_label='Centimeters'):
#         shifted_cmap = plotting.make_shifted_cmap(img, 'seismic')
#         plt.imshow(img, cmap=shifted_cmap)
#         cbar = plt.colorbar()
#         cbar.set_label(cbar_label)
#         plt.title(title)
#         plt.show(block=True)
#
#     logger.info("Using {} for igram path".format(igram_path))
#     logger.info("Using {}, {} for reference row, col to shift stack".format(row, col))
#
#     if not os.path.exists(igram_path):
#         logger.error("No directory {}".format(igram_path))
#         return
#
#     subset_dir = os.path.join(igram_path, 'subset')
#     utils.mkdir_p(subset_dir)
#     logger.info("Making subset directory for igrams in stack: {}".format(subset_dir))
#
#     for f in glob.glob(subset_dir + "*.unw"):
#         os.remove(f)
#
#     # intlist = read_intlist(igram_path)
#     geolist = read_geolist(igram_path)
#
#     geolist2 = np.array(geolist[9:-16])
#     print("List of .geo dates:")
#     pprint.pprint(list((idx, g) for idx, g in enumerate(geolist2)))
#
#     # timediffs = find_time_diffs(geolist)
#
#     pairs2 = [(d1.strftime("%Y%m%d"), d2.strftime("%Y%m%d"))
#               for d1, d2 in zip(geolist2[:-len(geolist2) // 2], geolist2[len(geolist2) // 2:])]
#     pairs2_names = ['_'.join(p) + '.unw' for p in pairs2]
#     print("Using unwrapped igrams:")
#     print(pprint.pformat(pairs2_names))
#     num_igrams = len(pairs2)
#
#     copyfile(os.path.join(igram_path, 'dem.rsc'), os.path.join(subset_dir, 'dem.rsc'))
#     for n in pairs2_names:
#         src = os.path.join(igram_path, n)
#         dest = os.path.join(subset_dir, n)
#         copyfile(src, dest)
#
#     unw_stack = sario.load_stack(directory=subset_dir, file_ext='.unw')
#     unw_stack = np.stack(remove_ramp(layer) for layer in unw_stack)
#
#     # Pick reference point and shift
#     unw_shifted = shift_stack(unw_stack, 100, 100, window=9, window_func=np.mean)
#     stack_mean = np.mean(unw_shifted, axis=0)
#
#     start_dt = _parse(pairs2[0][0])
#     end_dt = _parse(pairs2[-1][1])
#     total_days = (end_dt - start_dt).days
#     print("Start date:", start_dt.date())
#     print("End date:", end_dt.date())
#     print("Total Days:", total_days)
#
#     min_diff = np.min(stack_mean)
#     max_diff = np.max(stack_mean)
#     # Reshift again so that there is only uplift
#     stack_mean = stack_mean - max_diff
#     print("Min phase val, max val of the averaged stack")
#     print(min_diff, max_diff)
#     print("Min cm val, max cm val of the averaged deform")
#     print(PHASE_TO_CM * min_diff, PHASE_TO_CM * max_diff)
#     largest_phase_diff = min_diff if -1 * min_diff > max_diff else max_diff
#
#     print('largest mean phase diff:', largest_phase_diff)
#     print('in cm:', PHASE_TO_CM * largest_phase_diff)
#     print('=======' * 10)
#     plot_deformation(stack_mean, title='avged stack (in phase)', cbar_label='radians')
#
#     title = "Stack avg'ed deform. from {} to {}".format(start_dt.date(), end_dt.date())
#     plot_deformation(PHASE_TO_CM * stack_mean, title=title)
#
#     # Account for the different time periods in the igrams
#     tds = [_parse(d2) - _parse(d1) for d1, d2 in pairs2]
#     td_days = [d.days for d in tds]
#     print("Igram intervals (in days):")
#     print(td_days)
#     # unw_normed = np.stack([layer / days for layer, days in zip(unw_stack, td_days)])
#     unw_normed_shifted = np.stack([layer / days for layer, days in zip(unw_shifted, td_days)])
#
#     print("Diff in max phase:")
#     print(total_days * (np.max(unw_normed_shifted.reshape(
#       (num_igrams, -1)), axis=1) - np.min(unw_normed_shifted.reshape((num_igrams, -1)), axis=1)))
#     print("Converted to CM:")
#     print(total_days * (np.max(unw_normed_shifted.reshape(
#         (num_igrams, -1)) * PHASE_TO_CM, axis=1) - np.min(
#             unw_normed_shifted.reshape((num_igrams, -1)) * PHASE_TO_CM, axis=1)))
