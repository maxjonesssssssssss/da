import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class LogMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        String[] parts = value.toString().split(" ");
        if (parts.length >= 3) {
            String level = parts[2].toUpperCase();
            if (level.equals("INFO") || level.equals("ERROR") ||
                level.equals("WARNING") || level.equals("DEBUG")) {
                context.write(new Text(level), new IntWritable(1));
            }
        }
    }
}
