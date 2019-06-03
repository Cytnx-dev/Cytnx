{

    Tensor A({3,4,5});

    //move the tensor to different device by creating a clone object
    Tensor B = A.to(Device.cuda+0);
    cout << B.device_str() << endl;
    cout << A.device_str() << endl;    

    // move the instance tensor to different device
    A.to_(Device.cuda+0);
    cout << A.device_str() << endl;

}
