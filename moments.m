function moments = moments(d,box)
    n = size(box,2);
    dv = monpowers(n,d);
    moments = zeros(size(dv,1),1);
    for i = 1:numel(moments)
        moments(i) = prod((box(2,:).^(dv(i,:)+1) - box(1,:).^(dv(i,:)+1)) ./ (dv(i,:)+1));
    end
end

