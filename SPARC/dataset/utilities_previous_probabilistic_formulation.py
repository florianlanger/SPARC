import torch

def get_device(a):

    if a.is_cuda:
        gpu_n = a.get_device()
        device = torch.device("cuda:{}".format(gpu_n))
    else:
        device = torch.device("cpu")
    return device

def sample_points_from_lines(lines,points_per_line):

    device = get_device(lines)

    n_lines = lines.shape[0]
    lines = torch.repeat_interleave(lines,points_per_line,dim=0)
    interval = torch.linspace(0,1,points_per_line).repeat(n_lines).to(device)
    interval = interval.unsqueeze(1).repeat(1,2)
    points = lines[:,:2] + (lines[:,2:4]-lines[:,:2]) * interval
    points = points.view(n_lines,points_per_line,2)
    return points

def lines_3d_to_pixel(lines_3D,R,T,f,sw,img_size_reshaped,w):
    # need to project 3d line into 2d
    # then have two points for each line and can do whatever want
    # print('lines_3D,lines_2D,R,T,B',lines_3D.shape,lines_2D.shape,R.shape,T.shape,B.shape)
    lp1 = lines_3D[:,:3]
    lp2 = lines_3D[:,:3] + lines_3D[:,3:6]
    # need to normalise by dividing by z (maybe then neeed to multiply by f, check B z value)
    cc1 = torch.transpose(torch.matmul(R,torch.transpose(lp1,-1,-2)),-1,-2) + T
    cc2 = torch.transpose(torch.matmul(R,torch.transpose(lp2,-1,-2)),-1,-2) + T
    nc1 = cc1 / (torch.abs(cc1[:,2:3]) / f)
    nc2 = cc2 / (torch.abs(cc2[:,2:3]) / f)
    # print('nc1',nc1[:3])
    pix1_3d = pixel_bearing_to_pixel(nc1,w,sw,img_size_reshaped,f)
    pix2_3d = pixel_bearing_to_pixel(nc2,w,sw,img_size_reshaped,f)
    return pix1_3d,pix2_3d


def pixel_bearing_to_pixel(pb,w,sw,img_size,f):
    # img size needs to have same first dimension as pb
    assert img_size.shape[0] == pb.shape[0]
    assert len(img_size.shape) == 2
    assert img_size.shape[1] == 2
    assert len(pb.shape) == 2
    assert pb.shape[1] == 3
    # assert (torch.abs(pb[:,2] - f) < 0.001).all(),(pb[:,2],f)
    pixel = - pb[:,:2] * w/sw + img_size / 2
    # pixel = pixel[:,::-1]

    # px = - pb[:,0] * w/sw + w/2
    # print('px',px)
    return pixel

def pixel_bearing_to_pixel_from_calibration(pb,w,K):
    # img size needs to have same first dimension as pb
    assert len(pb.shape) == 2
    assert pb.shape[1] == 3

    fs = torch.Tensor([K[0,0],K[1,1]])
    cs = torch.Tensor([K[0,2],K[1,2]])
    # assert (torch.abs(pb[:,2] - f) < 0.001).all(),(pb[:,2],f)
    pixel = - pb[:,:2] * 2 * fs / w + cs
    # pixel = pixel[:,::-1]

    # px = - pb[:,0] * w/sw + w/2
    # print('px',px)
    return pixel

def create_all_possible_combinations_2(a,b):

    n1 = a.shape[0]
    n2 = b.shape[0]

    repeats_a = (1,) + (len(a.shape) - 1) * (1,)
    repeats_b = (n1,) + (len(b.shape) - 1) * (1,)

    batched_a = torch.repeat_interleave(a.repeat(repeats_a), n2, dim=0)
    batched_b = torch.repeat_interleave(b.repeat(repeats_b),1,dim=0)

    return batched_a,batched_b

def create_all_possible_combinations_2_dimension_1(a,b):

    n1 = a.shape[1]
    n2 = b.shape[1]

    repeats_a = (1,1,) + (len(a.shape) - 2) * (1,)
    repeats_b = (1,n1,) + (len(b.shape) - 2) * (1,)

    batched_a = torch.repeat_interleave(a.repeat(repeats_a), n2, dim=1)
    batched_b = torch.repeat_interleave(b.repeat(repeats_b),1,dim=1)

    return batched_a,batched_b