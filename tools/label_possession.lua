-- Possession labelling for mpv: press h / a / d where each stretch starts, x to close the last one.
-- Appends "start,end,label" rows, in tactical-video frames, to the file tools/make_validation_gt.py
-- reads. Output file: --script-opts=label-out=<path> (default validation/intervals.csv).
local fps = 25
local path = mp.get_opt("label-out") or "validation/intervals.csv"
local current, since = nil, nil

local function frame()
    return math.floor(mp.get_property_number("time-pos", 0) * fps + 0.5)
end

local function write_row(stop)
    local f = io.open(path, "r")
    local fresh = (f == nil) or (f:read("*l") == nil)
    if f then f:close() end
    f = assert(io.open(path, "a"))
    if fresh then f:write("start,end,label\n") end
    f:write(string.format("%d,%d,%s\n", since, stop, current))
    f:close()
end

local function mark(label)
    local now = frame()
    if current and now <= since then
        mp.osd_message("not saved: frame " .. now .. " is before the open stretch's start " .. since, 3)
        return
    end
    if current then write_row(now) end
    current, since = label, now
    mp.osd_message(label and (label .. " from frame " .. now) or ("closed at frame " .. now), 2)
end

mp.add_forced_key_binding("h", "label-home", function() mark("H") end)
mp.add_forced_key_binding("a", "label-away", function() mark("A") end)
mp.add_forced_key_binding("d", "label-dead", function() mark("D") end)
mp.add_forced_key_binding("x", "label-close", function() mark(nil) end)
