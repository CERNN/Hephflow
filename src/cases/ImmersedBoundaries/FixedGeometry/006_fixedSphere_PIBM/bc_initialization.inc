    if (z == (NZ - 1)) // F OUTLET
    {
        nodeType = (FRONT + BC_OUTFLOW);      
    }else if  (z == 0) // B
    {
        nodeType = (BACK  + BC_VELOCITY_WALL + BC_VELOCITY_INDEX_0);
    }
    else
    {        
        nodeType = BULK;   
    }