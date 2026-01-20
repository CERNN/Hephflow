    if (y == (NY - 1)) // F OUTLET
    {
        nodeType = (NORTH + BC_OUTFLOW);      
    }else if  (z == 0) // B
    {
        nodeType = (BACK  + BC_VELOCITY_WALL + BC_VELOCITY_INDEX_0);
    }
    else
    {        
        nodeType = BULK;   
    }