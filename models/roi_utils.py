import tortch

def get_ROI_grid(self, roi, src_size, dst_size, scale=1.):
        # scale height and width
        ry, rx, rh, rw = roi[:,0], roi[:,1], scale * roi[:,2], scale * roi[:,3]
        
        # convert ti minmax  
        ymin = ry - rh/2.
        ymax = ry + rh/2.
        xmin = rx - rw/2.
        xmax = rx + rw/2.
        
        h, w = src_size[0], src_size[1] 
        # theta
        theta = ToCudaVariable([torch.zeros(roi.size()[0],2,3)])[0]
        theta[:,0,0] = (xmax - xmin) / (w - 1)
        theta[:,0,2] = (xmin + xmax - (w - 1)) / (w - 1)
        theta[:,1,1] = (ymax - ymin) / (h - 1)
        theta[:,1,2] = (ymin + ymax - (h - 1)) / (h - 1)

        #inverse of theta
        inv_theta = ToCudaVariable([torch.zeros(roi.size()[0],2,3)])[0]
        det = theta[:,0,0]*theta[:,1,1]
        adj_x = -theta[:,0,2]*theta[:,1,1]
        adj_y = -theta[:,0,0]*theta[:,1,2]
        inv_theta[:,0,0] = w / (xmax - xmin) 
        inv_theta[:,1,1] = h / (ymax - ymin) 
        inv_theta[:,0,2] = adj_x / det
        inv_theta[:,1,2] = adj_y / det
        # make affine grid
        fw_grid = F.affine_grid(theta, torch.Size((roi.size()[0], 1, dst_size[0], dst_size[1])))
        bw_grid = F.affine_grid(inv_theta, torch.Size((roi.size()[0], 1, src_size[0], src_size[1])))
        return fw_grid, bw_grid, theta
