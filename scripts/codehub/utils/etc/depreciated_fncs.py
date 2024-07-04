def test_input_data(args,files,start_times,end_times):
    
    # Get the pathing to the excluded data
    if args.exclude == None:
        exclude_path = args.outdir+"excluded.txt"
    else:
        exclude_path = args.exclude

    # Get the files to use and which to save
    good_index = []
    bad_index  = []
    if os.path.exists(exclude_path):
        excluded_files = PD.read_csv(exclude_path)['file'].values
        for idx,ifile in enumerate(files):
            if ifile not in excluded_files:
                good_index.append(idx)
    else:
        # Confirm that data can be read in properly
        excluded_files = []
        for idx,ifile in enumerate(files):
            DLT  = data_loader_test()
            flag = DLT.edf_test(ifile)
            if flag[0]:
                good_index.append(idx)
            else:
                excluded_files.append([ifile,flag[1]])
        excluded_df = PD.DataFrame(excluded_files,columns=['file','error'])
        if not args.debug:
            excluded_df.to_csv(exclude_path,index=False)
    return files[good_index],start_times[good_index],end_times[good_index]


        # Consolidate the metadata for failed chunks and successful chunks, and squeeze the successful object to match output list
        ### There is an ocassional bug in how keys get handled. For now, blocking this code out and recommending people do not use data viability.
        """
        metadata_copy = self.metadata.copy()
        bad_metadata_keys = np.setdiff1d(list(self.metadata.keys()),self.output_meta)
        if bad_metadata_keys.size > 0:
            bad_metadata = self.metadata[bad_metadata_keys]
        else:
            bad_metadata = {}
        self.metadata = {}
        for idx,ikey in enumerate(self.output_meta):
            self.metadata[idx] = metadata_copy.pop(ikey)
        """